"""CLEAN 16-level scaling study (2026-10-22): shared constants.
Fresh formula bases, disjoint from every old base (31e9 TRB, 32e9 TEB, 33e9 VAB, 34e9 screen, 36e9, 37e9).
seed = base + 1e7*n + k (sim.instances convention), so bases 1e9 apart never collide for n<=20.
Hardware family: 16-level cavities (4 bits each), chain of K=n/4 cavities. compiler.LAYOUTS[20] is [5,5,5,5]
(32-level family) in the frozen compiler; we override it IN MEMORY ONLY to [4,4,4,4,4] (no file edits)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "v3", "native"), os.path.join(ROOT, "v3", "openevolve", "net")):
    if p not in sys.path: sys.path.insert(0, p)
import compiler as CP
CP.LAYOUTS[20] = [4, 4, 4, 4, 4]
for n in (8, 12, 16, 20): assert CP.LAYOUTS[n] == [4] * (n // 4)
TRB, TEB, VAB = 51e9, 52e9, 53e9
TEST_N = [8, 12, 16, 20]; NTEST = 100; VAL_N = 12; NVAL = 200
RES = os.path.join(ROOT, "results", "clean16"); os.makedirs(RES, exist_ok=True)
