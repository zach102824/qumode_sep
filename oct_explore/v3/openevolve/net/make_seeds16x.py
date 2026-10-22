"""build the 3 net16x island seeds (+ wildcard copy) from net16r bodies: zero-initialised pair-unitary head => seed == parent at init."""
import re, sys
HEAD = {
"reencode": '''# ---------------- PAIR-UNITARY HEAD (net16x island: reencode) ----------------
# Per-layer learned bit-to-cavity RELABEL realised by pair permutation unitaries that CONJUGATE the in-cavity mixer:
# slot 1 (between cost phase and mixer) U = exp(-i H_l), slot 2 (after mixer) U^dag = exp(+i H_l)  => the layer's mixer acts in a
# relabelled encoding (e.g. "flip bit i of A and bit j of B together" after a CNOT relabel A_i -> B_j), and the state is mapped back to
# the original encoding before the next cost phase / readout. H_l = sum_k th[l,k] * (pi/2)(I - P_k), P_k in {CNOT relabels, bit SWAPs}: th=1 is the exact
# permutation P_k (Gray-type relabel / bit moved to the other cavity), 0 < th < 1 a relaxed (still unitary) relabel. th zero-initialised => exactly the parent circuit.
PAIR_SLOTS = (1, 2)
def _bits16(i, j): return (i >> (3 - j)) & 1
def _cnot_gen():
    """48 generators (pi/2)(I - P): CNOT relabels A_i -> B_j, B_i -> A_j and bit SWAPs A_i <-> B_j on the 256 joint levels (index 16*a + b)."""
    G = []
    for d in (0, 1):
        for i in range(4):
            for j in range(4):
                P = np.zeros((256, 256), np.float32)
                for a in range(16):
                    for b in range(16):
                        if d == 0: a2, b2 = a, (b ^ (8 >> j)) if _bits16(a, i) else b
                        else: a2, b2 = ((a ^ (8 >> j)) if _bits16(b, i) else a), b
                        P[16 * a2 + b2, 16 * a + b] = 1.0
                G.append((np.pi / 2) * (np.eye(256, dtype=np.float32) - P))
    for i in range(4):   # + 16 exact bit SWAPs A_i <-> B_j (Opus seed design)
        for j in range(4):
            P = np.zeros((256, 256), np.float32)
            for a in range(16):
                for b in range(16):
                    ai, bj = _bits16(a, i), _bits16(b, j)
                    a2 = (a & ~(8 >> i)) | ((8 >> i) if bj else 0); b2 = (b & ~(8 >> j)) | ((8 >> j) if ai else 0)
                    P[16 * a2 + b2, 16 * a + b] = 1.0
            G.append((np.pi / 2) * (np.eye(256, dtype=np.float32) - P))
    return jnp.asarray(np.stack(G))
PAIR_G = _cnot_gen()
def pair_init(rng): return dict(pth=jnp.zeros((4, PAIR_G.shape[0]), jnp.float32))
def pair_outputs(p):
    Hl = jnp.einsum("lk,kij->lij", p["pth"], PAIR_G)                   # (L,256,256) real symmetric
    pHr = jnp.stack([Hl, -Hl], 1)[:, :, None]                         # (L, S=2, A=1 (same relabel on every pair), 256, 256)
    return dict(pHr=pHr, pslots=PAIR_SLOTS)
''',
"nondiag": '''# ---------------- PAIR-UNITARY HEAD (net16x island: nondiag) ----------------
# NON-DIAGONAL pair mixer between neighbouring cavities, slot 2 (after the in-cavity SNAP-D-SNAP mixer) of every layer:
# H_l = sum_k th[l,k] G_k with G = { X_A(i) X_B(j), Y_A(i) Y_B(j) : i,j in 0..3 } (correlated two-bit flips / bit flip-flop
#       "beam-splitter" hopping of a bit excitation across the cavity boundary).
# Same generator on every neighbour pair (A=1) => size-free and cheap (one 256x256 expm per layer). th zero-init => parent circuit.
PAIR_SLOTS = (2,)
def _nondiag_gen():
    """X_A(i)X_B(j) and Y_A(i)Y_B(j) (both real): correlated / flip-flop two-bit moves across the boundary. (Bosonic a^dag b was dropped:
    it conserves total photon number and mixes bits through integer carries - Opus seed review.)"""
    X = lambda j: np.array([[1.0 if (r ^ c) == (8 >> j) else 0.0 for c in range(16)] for r in range(16)], np.complex64)
    Y = lambda j: np.array([[(1j if not (c & (8 >> j)) else -1j) if (r ^ c) == (8 >> j) else 0.0 for c in range(16)] for r in range(16)], np.complex64)
    Gr = [np.kron(X(i), X(j)).real for i in range(4) for j in range(4)] + [np.kron(Y(i), Y(j)).real for i in range(4) for j in range(4)]
    Gr = np.stack(Gr).astype(np.float32) / 8.0
    return jnp.asarray(Gr), jnp.zeros_like(jnp.asarray(Gr[:1]))
PAIR_GR, PAIR_GI = _nondiag_gen()
def pair_init(rng): return dict(pth=jnp.zeros((4, PAIR_GR.shape[0]), jnp.float32))
def pair_outputs(p):
    Hr = jnp.einsum("lk,kij->lij", p["pth"], PAIR_GR)
    return dict(pHr=Hr[:, None, None], pslots=PAIR_SLOTS)  # (L,S=1,A=1,256,256)  # (L,S=1,A=1,256,256)
''',
"grover": '''# ---------------- PAIR-UNITARY HEAD (net16x island: grover) ----------------
# Grover-like REFLECTIONS built from local oracles: the oracle is the cavity-local clause-rank phase (pi on the capped set of levels that
# pass most in-scope clauses, see native()), the DIFFUSION is a pair reflection about the uniform state of each neighbour pair:
# slot 2 (after the mixer) H_l = th[l,0] * pi |s_AB><s_AB| + th[l,1] * pi (|s_A><s_A| x I + I x |s_B><s_B|)  (th=1: exact reflection(s)).
# Same on every neighbour pair (A=1). th zero-init => parent circuit. beta can learn to hand the mixing role over to the reflections.
PAIR_SLOTS = (2,)
def _grover_gen():
    sA = np.full((16, 1), 0.25, np.float32); PA = sA @ sA.T; I = np.eye(16, dtype=np.float32)
    sAB = np.full((256, 1), 1.0 / 16, np.float32)
    return jnp.asarray(np.stack([np.pi * (sAB @ sAB.T), np.pi * (np.kron(PA, I) + np.kron(I, PA))]))
PAIR_G = _grover_gen()
def pair_init(rng): return dict(pth=jnp.zeros((4, PAIR_G.shape[0]), jnp.float32))
def pair_outputs(p):
    return dict(pHr=jnp.einsum("lk,kij->lij", p["pth"], PAIR_G)[:, None, None], pslots=PAIR_SLOTS)
'''}
MANDATE = {"reencode": "reencode - per-layer learned bit-to-cavity relabel (pair permutation unitaries conjugating the mixer; mapped back each layer)",
           "nondiag": "nondiag - non-diagonal beam-splitter-like pair-unitary mixer between neighbouring cavities (correlated cross-boundary flips)",
           "grover": "grover - Grover-like reflections: local clause-rank oracles + pair diffusion reflections (pair unitaries)"}
def build(src, fam):
    src = re.sub(r"# FAMILY: [^\n]*", "# FAMILY: " + MANDATE[fam], src, count=1)
    src = src.replace("def apply(p, xe, xo, inc, sgn, V, place=None):", HEAD[fam] + "\ndef apply(p, xe, xo, inc, sgn, V, place=None):", 1)
    src = src.replace("p = init(rng); p.update(native_init(rng))", "p = init(rng); p.update(native_init(rng)); p.update(pair_init(rng))", 1)
    # model output: add shared pair outputs (training) ; predict: shared keys not chunk-concatenated
    src = src.replace("    return out\n\ndef train", "    out.update(pair_outputs(p))\n    return out\n\ndef train", 1)
    old = "        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}"
    assert old in src
    src = src.replace(old, "        sh = pair_outputs(pf)\n        return dict({k: np.concatenate([r[k] for r in res]) for k in res[0] if k not in sh}, **{k: (v if k == 'pslots' else np.asarray(v)) for k, v in sh.items()})", 1)
    src = src.replace("            res.append({k: np.nan_to_num(np.asarray(v)) for k, v in out.items()})", "            res.append({k: np.nan_to_num(np.asarray(v)) for k, v in out.items() if k not in ('pHr', 'pHi', 'pslots')})", 1)
    return src
if __name__ == "__main__":
    B10 = open("runs/net16r/checkpoints/checkpoint_10/best_program.py").read(); B110 = open("runs/net16r/checkpoints/checkpoint_110/best_program.py").read()
    for i, (fam, body) in enumerate((("reencode", B10), ("grover", B110), ("nondiag", B10))):
        s = build(body, fam); open(f"net/seeds16x/{i+1}_{fam}.py", "w").write(s); print(fam, "ok", len(s))
