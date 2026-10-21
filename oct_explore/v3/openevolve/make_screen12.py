"""Generate the n=12 STAGE-1 screening set for net16r (30 formulas, R=4 placements, exact ground states via frozen make_set).
Base seed SCB = 34e9 + 1e8*cycle (train TRB=31e9, test TEB=32e9, selection VAB=33e9; make_set seeds are base + 1e7*n + k, k < 1e7,
so bases differing by >= 1e8 can never collide). Asserts no seed and no formula (V,SG) overlap with train / VAB / TEB sets on disk.
usage: python3.13 make_screen12.py [cycle]   -> runs/datacache/n12_c30_b<SCB>_R4.npz, prints the base."""
import os, sys, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "net"))
import evaluator_net as EV, frozen_circuit as FC
def base_of(cycle): return 34e9 + 1e8 * int(cycle)
def make(cycle=0, count=30, n=12):
    b = base_of(cycle); assert b not in (FC.TRB, FC.TEB, FC.VAB)
    D = EV.get_set(n, count, b, EV.R)
    key = lambda V, SG: (V.tobytes(), SG.tobytes())
    mine = {key(V, S) for V, S in zip(D["V"], D["SG"])}; myseeds = set(int(s) for s in D["seed"])
    for f in glob.glob(os.path.join(EV.CACHE, "n*_c*_b*_R4.npz")):
        if f"_b{int(b)}_" in f: continue
        z = np.load(f, allow_pickle=True)
        assert not (myseeds & set(int(s) for s in z["seed"])), f"seed overlap with {f}"
        if z["V"].shape[1:] == D["V"].shape[1:]:
            assert not (mine & {key(V, S) for V, S in zip(z["V"], z["SG"])}), f"formula overlap with {f}"
    assert all(len(s) > 0 for s in D["sol"]), "every screen formula needs an exact ground state"
    return b, D
if __name__ == "__main__":
    b, D = make(int(sys.argv[1]) if len(sys.argv) > 1 else 0); print(int(b), D["V"].shape, "ok: no overlap with train/VAB/TEB")
