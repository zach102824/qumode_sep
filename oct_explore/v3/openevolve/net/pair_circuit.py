"""net16x EXTENSION of the frozen circuit: ONE new gate type, the neighbour PAIR UNITARY U = exp(-i H) on two adjacent cavities
(16x16 = 256 joint Fock states, joint index = 16*i_a + i_b, i_c = integer of cavity c's bits MSB first, same as cav_idx / dB tables).
H is any Hermitian 256x256 (time t folded into H). frozen_circuit.py / evaluator_net.py are NOT modified: this file reuses their
compiler projection, mixers, prep and readout verbatim and only inserts the pair-unitary slots. With no pair unitaries (or H=0)
circuit_px == frozen circuit_p exactly (unit-tested in test_pair16x.py).
Slots per layer l: slot 0 = BEFORE the cost phase, slot 1 = BETWEEN cost phase and mixer, slot 2 = AFTER the mixer;
within a slot pairs a=0..C-2 in order. => at most 3*(C-1) pair unitaries per layer (n=16: 9). Program outputs (optional):
  pslots   : which slots are used, a tuple/array of distinct ints from {0,1,2} (default (0,1,2)); S = len(pslots).
             Unused slots cost nothing (no expm). e.g. pslots=(1,2) with U_mid = P, U_post = P^dag conjugates the mixer.
  pHr, pHi : SHARED generator (real/imag part), shape (L,S,A,256,256), A = C-1 (one per pair) or A = 1 (same on every pair; cheapest).
  pG, pGi  : STRUCTURED basis of K<=16 generators (real/imag part), shape (K,256,256), shared;
  pc       : per-entry coefficients (F,R,L,S,C-1,K)  ->  H_entry = sum_k pc_k (pG_k + i pGi_k)  (+ shared pHr/pHi block).
  The evaluator Hermitises every block: H = (A + A^dag)/2. A block that is all zero is the identity and is not counted.
  (A full per-entry 256x256 generator is NOT accepted: (F,R,L,2,C-1,256,256) does not fit the memory cap.)
Cost note: each nonzero per-entry block costs one 256x256 complex expm (~20 matmuls); shared generators are ~F*R x cheaper."""
import numpy as np, jax, jax.numpy as jnp
import frozen_circuit as FC
from frozen_circuit import NS, CP
PD = 256
def herm(Hr, Hi=None):
    A = jnp.asarray(Hr, jnp.float32).astype(jnp.complex64)
    if Hi is not None: A = A + 1j * jnp.asarray(Hi, jnp.float32).astype(jnp.complex64)
    return 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))
def pair_unitary(H):
    """U = exp(-i H) for Hermitian H (..., 256, 256); differentiable (Pade expm)."""
    return jax.scipy.linalg.expm(-1j * H)
def apply_pair(U, psi, a, C):
    """U (256,256) or (B,256,256) on cavities (a, a+1) of psi (B, N_0..N_{C-1}) with N=16."""
    let = "defghijk"[:C]; o1, o2 = let[a], let[a + 1]; out = let.replace(o1, "y").replace(o2, "z")
    if U.ndim == 2: return jnp.einsum(f"yz{o1}{o2},b{let}->b{out}", U.reshape(16, 16, 16, 16), psi)
    return jnp.einsum(f"byz{o1}{o2},b{let}->b{out}", U.reshape(-1, 16, 16, 16, 16), psi)
def _slot(Us, l, s, psi, C, shared, slots=(0, 1, 2)):
    if Us is None or s not in slots: return psi
    j = list(slots).index(s); A = Us.shape[-3]
    for a in range(C - 1):
        aa = a if A > 1 else 0
        U = Us[l, j, aa] if shared else Us[:, l, j, aa]
        psi = apply_pair(U, psi, a, C)
    return psi
def slots_of(pair):
    if not pair or pair.get("pslots") is None: return (0, 1, 2)
    try: t = tuple(int(x) for x in np.asarray(pair["pslots"]).reshape(-1))
    except Exception: raise ValueError("pslots must be a static python tuple / numpy constant (not a traced jax array)"); assert len(set(t)) == len(t) and all(x in (0, 1, 2) for x in t), f"pslots {t}"
    return t
KMAX = 16
def gen_H(pair, B, n):
    """pair dict (pHr,pHi shared; pG,pGi,pc structured with pc (B,L,2,C-1,K)) -> Hermitian H (L,2,C-1,256,256) shared or (B,...) or None."""
    if not pair: return None
    C = len(CP.LAYOUTS[n]); H = None; S = len(slots_of(pair))
    assert all(k == 4 for k in CP.LAYOUTS[n]), "pair unitary needs 4-bit cavities"
    if pair.get("pHr") is not None:
        H = herm(pair["pHr"], pair.get("pHi")); assert H.shape[:2] == (FC.L, S) and H.shape[2] in (1, C - 1) and H.shape[3:] == (PD, PD), f"shared pHr shape {H.shape} != (L,S,C-1 or 1,256,256)"
    if pair.get("pc") is not None:
        G = herm(pair["pG"], pair.get("pGi")); K = G.shape[0]; assert G.shape == (K, PD, PD) and K <= KMAX, f"pG shape {G.shape} (K<={KMAX})"
        c = jnp.asarray(pair["pc"], jnp.float32); assert c.shape == (B, FC.L, S, C - 1, K), f"pc shape {c.shape} != {(B, FC.L, S, C - 1, K)}"
        He = jnp.einsum("blsak,kij->blsaij", c.astype(jnp.complex64), G); H = He if H is None else He + H[None]
    return H
def unitaries(pair, B, n):
    H = gen_H(pair, B, n)
    if H is None: return None, True
    return pair_unitary(H), H.ndim == 5
def count_pairs(pair, C, tol=1e-6):
    """number of nonzero pair-unitary blocks per circuit (max over entries); a broadcast (A=1) block counts C-1 times."""
    if not pair: return 0
    nz = None
    if pair.get("pHr") is not None:
        A = np.abs(np.asarray(pair["pHr"])) + (np.abs(np.asarray(pair["pHi"])) if pair.get("pHi") is not None else 0); nz = A.max(axis=(-1, -2)) > tol   # (L,S,A)
        nz = np.broadcast_to(nz, nz.shape[:2] + (C - 1,))
    if pair.get("pc") is not None:
        g = np.abs(np.asarray(pair["pG"])).max(axis=(1, 2)) + (np.abs(np.asarray(pair["pGi"])).max(axis=(1, 2)) if pair.get("pGi") is not None else 0)
        c = np.asarray(pair["pc"]); c = c.reshape(-1, *c.shape[-4:]); e = (np.abs(c) * g).max(-1) > tol   # (B,L,S,C-1)
        if nz is not None: e = e | nz[None]
        return int(e.sum(axis=(1, 2, 3)).max())
    return 0 if nz is None else int(nz.sum())
def circuit_px(n, gamma, beta, w, pv, VP, BB, PERM, S, nm, dS=None, dB=None, glob=False, pair=None, slots=None):
    """frozen circuit_p + pair-unitary slots. Returns p (B,)."""
    ks = CP.LAYOUTS[n]; B = S.shape[0]; C = len(ks)
    SL = tuple(slots) if slots is not None else slots_of(pair); pair = {k: v for k, v in (pair or {}).items() if k != "pslots"} or None
    if pair is not None: pair["pslots"] = SL
    Us, shared = unitaries(pair, B, n)
    pvc = jnp.take_along_axis(pv, PERM.astype(jnp.int32), axis=1)
    xb = jnp.asarray(FC.bits(n)); lit = xb[:, VP]; viol = jnp.prod((lit == BB[None].astype(jnp.int8)).astype(jnp.float32), -1)
    psi = NS.uniform_psi0(ks, B)
    for l in range(FC.L):
        psi = _slot(Us, l, 0, psi, C, shared, SL)
        Ew = nm * jnp.einsum("xbm,bm->bx", viol, gamma[:, l][:, None] + w[:, :, l])
        if not glob: Ew = FC.project_product_j(Ew, pvc, n, ks)
        if dS is not None:
            I = FC.cav_idx(n)
            for c in range(C): Ew = Ew + jnp.take(dS[:, l, c], jnp.asarray(I[c]), axis=1)
        if dB is not None:
            I = FC.cav_idx(n)
            for a in range(C - 1): Ew = Ew + jnp.take(dB[:, l, a].reshape(B, -1), jnp.asarray(I[a] * 32 + I[a + 1]), axis=1)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64); psi = _slot(Us, l, 1, psi, C, shared, SL)
        for c, k in enumerate(ks): psi = NS.apply_cav(FC.MIXK[k](beta[:, l]), psi, c, C)
        psi = _slot(Us, l, 2, psi, C, shared, SL)
    return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * S, 1)
def circuit_state(n, gamma, beta, w, pv, VP, BB, PERM, S, nm, dS=None, dB=None, pair=None, slots=None):
    """diagnostic: final (or after layer `upto`) state |psi|^2 (B, 2^n) chain order (evaluator-side only)."""
    ks = CP.LAYOUTS[n]; B = S.shape[0]; C = len(ks); SL = tuple(slots) if slots is not None else slots_of(pair); pair = {k: v for k, v in (pair or {}).items() if k != "pslots"} or None
    if pair is not None: pair["pslots"] = SL
    Us, shared = unitaries(pair, B, n)
    pvc = jnp.take_along_axis(pv, PERM.astype(jnp.int32), axis=1)
    xb = jnp.asarray(FC.bits(n)); lit = xb[:, VP]; viol = jnp.prod((lit == BB[None].astype(jnp.int8)).astype(jnp.float32), -1)
    psi = NS.uniform_psi0(ks, B); outs = []
    for l in range(FC.L):
        psi = _slot(Us, l, 0, psi, C, shared, SL)
        Ew = FC.project_product_j(nm * jnp.einsum("xbm,bm->bx", viol, gamma[:, l][:, None] + w[:, :, l]), pvc, n, ks)
        I = FC.cav_idx(n)
        if dS is not None:
            for c in range(C): Ew = Ew + jnp.take(dS[:, l, c], jnp.asarray(I[c]), axis=1)
        if dB is not None:
            for a in range(C - 1): Ew = Ew + jnp.take(dB[:, l, a].reshape(B, -1), jnp.asarray(I[a] * 32 + I[a + 1]), axis=1)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64); psi = _slot(Us, l, 1, psi, C, shared, SL)
        for c, k in enumerate(ks): psi = NS.apply_cav(FC.MIXK[k](beta[:, l]), psi, c, C)
        psi = _slot(Us, l, 2, psi, C, shared, SL); outs.append(jnp.abs(psi.reshape(B, -1)) ** 2)
    return outs
