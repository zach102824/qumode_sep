"""Nov9th leak-echo experiment core: instances, ED, Nov9th Fock simulator, QAOA, SPSA.

Provisional mathematical model (user has NOT confirmed the physical off-code definitions):
  D2 SNAP phase 0 on Fock f >= 8; D3 Uz (pair terms touching a leaked cavity act as identity),
  U0 sensitivity (whole product component unphased if any cavity leaked);
  D4 per-term penalised H_ext; D5 ideal number-resolved readout, ideal W_L on the code.
Encoding: binary Fock j=0..7 per cavity (Nov_3rd/PLAN.md L12, L33); bit r of f is (f >> r) & 1.
Grouping: interleaved, cavity a holds sites [a, a+m, a+2m], bit r <-> site a + r*m.
Qubit convention: site i = bit i of the integer index, Z|0> = +|0>.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

NU = 8
ALPHA_MAX = 0.30
NF_PAD = 32


# ---------------------------------------------------------------- instances

def make_J(seed: int, N: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0 / np.sqrt(N), size=N * (N - 1) // 2)
    J = np.zeros((N, N))
    t = 0
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = u[t]
            t += 1
    return J


def J_sha(J: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(J, dtype="<f8").tobytes()).hexdigest()


def upper(Q):
    iu = np.triu_indices(Q.shape[0], 1)
    return Q[iu]


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def packing(N):
    m = N // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def feature_K(J):
    K = J @ J
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def feature_M(J, pack):
    Mx = np.zeros_like(J)
    for a in range(len(pack)):
        for b in range(a + 1, len(pack)):
            pairs = [(i, j) for i in pack[a] for j in pack[b]]
            w = float(np.mean([J[i, j] for i, j in pairs]))
            for i, j in pairs:
                Mx[i, j] = Mx[j, i] = w
    return Mx


def hat(F, J):
    r = rms(upper(F))
    if r <= 0:
        raise RuntimeError("zero feature")
    return F * (rms(upper(J)) / r)


# ---------------------------------------------------------------- qubit space / ED

def z_table(N):
    x = np.arange(1 << N, dtype=np.int64)
    return np.stack([1.0 - 2.0 * ((x >> i) & 1) for i in range(N)])


def hzz_diag(J, z):
    N = J.shape[0]
    d = np.zeros(z.shape[1])
    for i in range(N):
        for j in range(i + 1, N):
            d += J[i, j] * z[i] * z[j]
    return d


def apply_sumX(psi, N):
    out = np.zeros_like(psi)
    idx = np.arange(psi.size)
    for i in range(N):
        out += psi[idx ^ (1 << i)]
    return out


def exact_gs(J, b):
    """Returns E0, gs (real, positive), E1, E0_odd. Evaluation only."""
    N = J.shape[0]
    z = z_table(N)
    d = hzz_diag(J, z)
    dim = 1 << N
    mask = dim - 1
    idx = np.arange(dim)

    def mv(v):
        v = np.asarray(v).reshape(-1)
        return d * v - b * apply_sumX(v, N)

    H = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    if dim <= 4096:
        rows, cols, vals = [idx], [idx], [d]
        for i in range(N):
            rows.append(idx)
            cols.append(idx ^ (1 << i))
            vals.append(np.full(dim, -b))
        Hs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
        w, V = np.linalg.eigh(Hs.toarray())
        E0, E1 = float(w[0]), float(w[1])
        g = V[:, 0]
        par = np.array([float(V[:, k] @ V[idx ^ mask, k]) for k in range(min(dim, 64))])
        odd = [k for k in range(len(par)) if par[k] < 0]
        E0_odd = float(w[odd[0]]) if odd else float("nan")
    else:
        w, V = eigsh(H, k=2, which="SA", tol=1e-12)
        o = np.argsort(w)
        E0, E1 = float(w[o[0]]), float(w[o[1]])
        g = V[:, o[0]]
        shift = 20.0 * N

        def mvo(v):
            v = np.asarray(v).reshape(-1)
            return mv(v) + shift * 0.5 * (v + v[idx ^ mask])

        Ho = LinearOperator((dim, dim), matvec=mvo, dtype=np.float64)
        wo, _ = eigsh(Ho, k=1, which="SA", tol=1e-10)
        E0_odd = float(wo[0])
    if g.sum() < 0:
        g = -g
    g = g / np.linalg.norm(g)
    resid = float(np.linalg.norm(mv(g) - E0 * g))
    return {"E0": E0, "E1": E1, "E0_odd": E0_odd, "gs": g, "resid": resid, "hzz": d}


def plus_fidelity(gs):
    return float(np.sum(gs) ** 2 / gs.size)


# ---------------------------------------------------------------- displacement

def ladder_gen(nf):
    """Hermitian X = i(a^dag - a) on nf levels."""
    a = np.diag(np.sqrt(np.arange(1, nf, dtype=np.float64)), 1)
    return 1j * (a.T - a)


class Disp:
    """True displacement D(alpha)=exp(alpha a^dag - alpha a), real alpha, built in nf+NF_PAD then
    restricted to nf. strict=True gives the strict-logical D8 = exp of the cropped 8x8 generator
    (NOT a displacement; diagnostic only)."""

    def __init__(self, nf, strict=False):
        self.nf = nf
        self.strict = strict
        big = NU if strict else nf + NF_PAD
        lam, V = np.linalg.eigh(ladder_gen(big))
        self.lam, self.V = lam, V

    def __call__(self, alpha):
        U = (self.V * np.exp(-1j * alpha * self.lam)) @ self.V.conj().T
        U = U.real  # real alpha -> real orthogonal
        return np.ascontiguousarray(U[: self.nf, : self.nf])


# ---------------------------------------------------------------- Nov9th simulator

def popcount8():
    return np.array([bin(f).count("1") for f in range(NU)])


def shell_phase(s, nf):
    """Phase vector theta_f for exp(-i theta_f); pad 0 above code (D2)."""
    p = popcount8()
    th = np.zeros(nf)
    th[:NU] = s[0] * p + s[1] * p * (p - 1) / 2.0 + s[2] * (p == 3)
    return th


class Nov9:
    """|+>^N -> [UcZ -> D(alpha_l)^{xC} -> S_l^{xC}]^L on the Fock tensor (nf,)*C."""

    def __init__(self, J, b, L, nf=24, ext="Uz", strict=False, features=("K", "M"), s3off=False):
        N = J.shape[0]
        assert N % 3 == 0
        self.N, self.C, self.m, self.L, self.b = N, N // 3, N // 3, L, float(b)
        self.strict = strict
        self.nf = NU if strict else nf
        self.ext = ext
        self.s3off = s3off
        self.J = J
        self.pack = packing(N)
        self.features = tuple(features)
        F = {"K": hat(feature_K(J), J), "M": hat(feature_M(J, self.pack), J)}
        self.Fhat = [F[k] for k in self.features]
        self.disp = Disp(self.nf, strict=strict)
        nf, C = self.nf, self.C
        f = np.arange(nf)
        self.cm = (f < NU).astype(float)
        self.zf = np.zeros((3, nf))
        for r in range(3):
            self.zf[r, :NU] = 1.0 - 2.0 * ((f[:NU] >> r) & 1)
        self.shape = (nf,) * C
        self.allcode = self._outer_prod([self.cm] * C)
        self.G = [self._pair_diag(Q) for Q in [J] + self.Fhat]
        if ext == "U0":
            self.G = [g * self.allcode for g in self.G]
        elif ext != "Uz":
            raise ValueError(ext)
        self.Ezz = self._pair_diag(J) + self._leak_pen(np.abs(J))
        self.EzzH0 = self._pair_diag(J) * self.allcode
        self.plus = np.zeros(self.shape, dtype=np.complex128)
        sl = tuple(slice(0, NU) for _ in range(C))
        self.sl = sl
        self.plus[sl] = 1.0 / math.sqrt(2.0 ** N)
        # code-block index -> qubit integer x
        grids = np.meshgrid(*[np.arange(NU)] * C, indexing="ij")
        x = np.zeros((NU,) * C, dtype=np.int64)
        for c in range(C):
            for r in range(3):
                x |= ((grids[c] >> r) & 1) << (c + r * self.m)
        self.code_to_x = x

    def site(self, c, r):
        return c + r * self.m

    def _bcast(self, v, c):
        sh = [1] * self.C
        sh[c] = self.nf
        return v.reshape(sh)

    def _outer_prod(self, vs):
        out = np.ones(self.shape)
        for c, v in enumerate(vs):
            out = out * self._bcast(v, c)
        return out

    def _pair_diag(self, Q):
        """sum_{i<j} Q_ij Zt_i Zt_j with zero-extended Zt (0 on leaked levels)."""
        out = np.zeros(self.shape)
        C = self.C
        for c in range(C):
            v = np.zeros(self.nf)
            for r in range(3):
                for rr in range(r + 1, 3):
                    v += Q[self.site(c, r), self.site(c, rr)] * self.zf[r] * self.zf[rr]
            out = out + self._bcast(v, c)
            for d in range(c + 1, C):
                Mcd = np.zeros((self.nf, self.nf))
                for r in range(3):
                    for rr in range(3):
                        Mcd += Q[self.site(c, r), self.site(d, rr)] * np.outer(self.zf[r], self.zf[rr])
                sh = [1] * C
                sh[c] = self.nf
                sh[d] = self.nf
                out = out + Mcd.reshape([self.nf if k in (c, d) else 1 for k in range(C)])
        return out

    def _leak_pen(self, Qabs):
        """sum over pairs touching a leaked cavity of |J_ij| (D4 per-term)."""
        out = np.zeros(self.shape)
        C = self.C
        lk = 1.0 - self.cm
        for c in range(C):
            w = sum(Qabs[self.site(c, r), self.site(c, rr)] for r in range(3) for rr in range(r + 1, 3))
            out = out + w * self._bcast(lk, c)
            for d in range(c + 1, C):
                w = sum(Qabs[self.site(c, r), self.site(d, rr)] for r in range(3) for rr in range(3))
                both = np.outer(self.cm, self.cm)
                out = out + w * (1.0 - both).reshape([self.nf if k in (c, d) else 1 for k in range(C)])
        return out

    # parameter layout: [gamma, theta_f..., u_1..u_L, s_{l,1..3 or 1..2}]
    def nparam(self):
        ns = 2 if self.s3off else 3
        return 1 + len(self.Fhat) + self.L + ns * self.L

    def unpack(self, x):
        nfeat = len(self.Fhat)
        gamma = x[0]
        theta = x[1 : 1 + nfeat]
        u = x[1 + nfeat : 1 + nfeat + self.L]
        ns = 2 if self.s3off else 3
        s = x[1 + nfeat + self.L :].reshape(self.L, ns)
        if self.s3off:
            s = np.column_stack([s, np.zeros(self.L)])
        return gamma, theta, ALPHA_MAX * np.tanh(u), s

    def sigma(self):
        nfeat = len(self.Fhat)
        ns = 2 if self.s3off else 3
        return np.concatenate([[0.10], np.full(nfeat, 0.25), np.full(self.L, 0.067), np.full(ns * self.L, 0.15)])

    def _apply_local(self, psi, M, c):
        nf = self.nf
        a = nf ** c
        bb = psi.size // (a * nf)
        out = np.matmul(M, psi.reshape(a, nf, bb))
        return out.reshape(self.shape)

    def state(self, x):
        gamma, theta, alpha, s = self.unpack(np.asarray(x, float))
        diag = self.G[0].copy()
        for k, th in enumerate(theta):
            diag += th * self.G[1 + k]
        phase = np.exp(-1j * gamma * diag)
        psi = self.plus.copy()
        for ell in range(self.L):
            psi = psi * phase
            D = self.disp(alpha[ell])
            for c in range(self.C):
                psi = self._apply_local(psi, D, c)
            ph = np.exp(-1j * shell_phase(s[ell], self.nf))
            psi = psi * self._outer_prod([ph] * self.C)
        return psi

    def xl_expect(self, psi):
        """sum_i <X^L_i P_{c(i)}> (logical X on the code block of its cavity)."""
        tot = 0.0
        for c in range(self.C):
            sub = np.take(psi, np.arange(NU), axis=c)
            for r in range(3):
                perm = np.arange(NU) ^ (1 << r)
                tot += float(np.real(np.vdot(sub, np.take(sub, perm, axis=c))))
        return tot

    def leak_per_cavity(self, prob):
        out = []
        for c in range(self.C):
            ax = tuple(k for k in range(self.C) if k != c)
            pc = prob.sum(axis=ax)
            out.append(float(pc[NU:].sum()))
        return out

    def energy_of_state(self, psi):
        prob = np.abs(psi) ** 2
        ezz = float(np.sum(prob * self.Ezz))
        xl = self.xl_expect(psi)
        lk = self.leak_per_cavity(prob)
        return ezz - self.b * xl + 3.0 * self.b * sum(lk)

    def energy(self, x):
        return self.energy_of_state(self.state(x))

    def diagnostics(self, x, gs):
        psi = self.state(x)
        prob = np.abs(psi) ** 2
        code = psi[self.sl]
        ncode = float(np.sum(np.abs(code) ** 2))
        g = gs[self.code_to_x]
        ov = complex(np.vdot(g, code))
        F = abs(ov) ** 2
        # H0 (leak scored 0): P_all H P_all
        ezz0 = float(np.sum(prob * self.EzzH0))
        xl0 = 0.0
        for c in range(self.C):
            for r in range(3):
                perm = np.arange(NU) ^ (1 << r)
                xl0 += float(np.real(np.vdot(code, np.take(code, perm, axis=c))))
        par = float(np.real(np.vdot(code, code[tuple(slice(None, None, -1) for _ in range(self.C))])))
        return {
            "E": self.energy_of_state(psi),
            "E_H0": ezz0 - self.b * xl0,
            "F": F,
            "p_leak": 1.0 - ncode,
            "F_postselected_not_physical": F / max(ncode, 1e-300),
            "parity": par,
            "norm": float(prob.sum()),
            "leak_per_cavity": self.leak_per_cavity(prob),
        }


# ---------------------------------------------------------------- QAOA

class QAOA:
    def __init__(self, J, b, p):
        self.N, self.p, self.b = J.shape[0], p, float(b)
        self.hzz = hzz_diag(J, z_table(self.N))
        self.plus = np.full(1 << self.N, 2.0 ** (-0.5 * self.N), dtype=np.complex128)
        self.idx = np.arange(1 << self.N)

    def nparam(self):
        return 2 * self.p

    def sigma(self):
        return np.full(2 * self.p, 0.10)

    def state(self, x):
        g, be = x[: self.p], x[self.p :]
        psi = self.plus.copy()
        N = self.N
        for ell in range(self.p):
            psi = psi * np.exp(-1j * g[ell] * self.hzz)
            c, s = math.cos(be[ell]), math.sin(be[ell])
            for i in range(N):
                v = psi.reshape(-1, 2, 1 << i)
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a0 - 1j * s * a1
                v[:, 1, :] = c * a1 - 1j * s * a0
                psi = v.reshape(-1)
        return psi

    def energy_of_state(self, psi):
        prob = np.abs(psi) ** 2
        ex = 0.0
        for i in range(self.N):
            ex += float(np.real(np.vdot(psi, psi[self.idx ^ (1 << i)])))
        return float(prob @ self.hzz) - self.b * ex

    def energy(self, x):
        return self.energy_of_state(self.state(x))

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = abs(np.vdot(gs, psi)) ** 2
        mask = (1 << self.N) - 1
        par = float(np.real(np.vdot(psi, psi[self.idx ^ mask])))
        return {"E": self.energy_of_state(psi), "F": float(F), "p_leak": 0.0, "parity": par,
                "norm": float(np.sum(np.abs(psi) ** 2))}


# ---------------------------------------------------------------- SPSA

class Budget:
    def __init__(self, f, cap):
        self.f, self.cap, self.n = f, cap, 0
        self.cat = {"calibration": 0, "perturb": 0, "candidate": 0, "monitor": 0}

    def __call__(self, x, cat):
        if self.n >= self.cap:
            raise RuntimeError("query cap exceeded")
        self.n += 1
        self.cat[cat] += 1
        e = float(self.f(x))
        if not np.isfinite(e):
            raise FloatingPointError("non-finite energy")
        return e


def spsa_start(B, x0, sig, rng, K, ncal=4, A=25.0, c=1.0, target=0.5, clip=2.0, avg_tail=64):
    d = x0.size
    y = np.zeros(d)
    gabs = np.zeros(d)
    for _ in range(ncal):
        De = rng.choice([-1.0, 1.0], size=d)
        ep = B(x0 + sig * c * De, "calibration")
        em = B(x0 - sig * c * De, "calibration")
        gabs += np.abs((ep - em) / (2 * c) * De)
    gabs /= ncal
    med = float(np.median(gabs))
    a = target * (A + 1.0) ** 0.602 / max(med, 1e-12)
    hist = []
    for k in range(K):
        ak = a / (k + 1.0 + A) ** 0.602
        ck = c / (k + 1.0) ** 0.101
        De = rng.choice([-1.0, 1.0], size=d)
        ep = B(x0 + sig * (y + ck * De), "perturb")
        em = B(x0 + sig * (y - ck * De), "perturb")
        g = (ep - em) / (2 * ck) * De
        y = y - np.clip(ak * g, -clip, clip)
        hist.append(y.copy())
    cands = [("last", y.copy()), ("polyak", np.mean(hist[-min(avg_tail, len(hist)):], axis=0) if hist else y.copy())]
    out = []
    for name, yy in cands:
        xx = x0 + sig * yy
        out.append((B(xx, "candidate"), name, xx))
    return out, {"a": a, "median_absg": med}


def spsa_run(f, starts, sig, rng_for_start, cap):
    """cap split equally across len(starts) starts; each start = 2*ncal + 2K + 2 queries."""
    ns = len(starts)
    per = cap // ns
    ncal = 4
    K = (per - 2 * ncal - 2) // 2
    assert 2 * ncal + 2 * K + 2 == per, (per, K)
    B = Budget(f, cap)
    allc = []
    meta = []
    for si, x0 in enumerate(starts):
        cands, m = spsa_start(B, np.asarray(x0, float), sig, rng_for_start(si), K, ncal=ncal)
        for e, name, xx in cands:
            allc.append((e, si, name, xx))
        meta.append(m)
    e, si, name, xx = min(allc, key=lambda t: t[0])
    return {"E_sel": e, "start": si, "cand": name, "x": xx, "queries": B.n, "query_cat": dict(B.cat),
            "K_per_start": K, "start_meta": meta, "all_cand_E": [t[0] for t in allc]}
