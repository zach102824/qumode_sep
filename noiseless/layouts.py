"""Named variable→slot layouts for the n=8 noiseless runners.

``identity``    legacy layout, class 0 (Z1→d, Z2→e, Z3–Z5→A2..A0, Z6–Z8→B2..B0), binary code.
``rule_best``   tier rule on the TRUE GS (results/LAYOUT_PATTERNS.md): permutation-only, binary
                code, each cavity's GS codeword 000 or 111 (Fock 0 or 7); deterministic
                tie-break (:func:`noiseless.encoding.rule_layout_for_bitstring`). Oracle.
``rule_bad``    TRUE GS at Fock levels in {2..5} on both cavities (avoid-rule violator), Fock
                3/4 preferred, deterministic. Oracle "worst-type" layout.
``screen_best`` per-Hamiltonian best class of the brute-force encoding screen (highest mean
                p(GS) over its 3 inits), read from noiseless/data/encoding_screen/*.jsonl.gz.
                Only for Hamiltonians whose screen data exists (H0–H9). Oracle; note the
                screen's inits 1–3 are the same seeds as trials 1–3 of run_u_sweep's jp/L=4
                scheme, so those trials carry selection bias.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

from noiseless.encoding import EncodingSpec, rule_layout_for_bitstring

LAYOUTS = ("identity", "rule_best", "rule_bad", "screen_best")
_REPO = Path(__file__).resolve().parents[1]
SCREEN_DATA_DIR = _REPO / "noiseless" / "data" / "encoding_screen"


@lru_cache(maxsize=64)
def screen_best_perm(ham_file: str, data_dir: str = str(SCREEN_DATA_DIR)) -> tuple[int, ...]:
    """perm of the class with the highest mean p(GS) in the encoding screen for ``ham_file``.

    Ties (equal mean to 1e-15) break by the smaller class_idx.
    """
    stem = Path(ham_file).name.replace(".npz", "")
    path = Path(data_dir) / f"{stem}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"no encoding-screen data for {ham_file} ({path})")
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    perms: dict[int, tuple[int, ...]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            c = int(r["class_idx"])
            sums[c] = sums.get(c, 0.0) + float(r["p_gs"])
            counts[c] = counts.get(c, 0) + 1
            perms[c] = tuple(int(v) for v in r["perm"])
    best = min(sums, key=lambda c: (-sums[c] / counts[c], c))
    return perms[best]


def layout_spec(name: str, ground_bitstring: str, ham_file: str | None = None) -> EncodingSpec:
    """EncodingSpec for a named layout (see module docstring)."""
    n = str(name).lower()
    if n == "identity":
        return EncodingSpec()
    if n == "rule_best":
        return rule_layout_for_bitstring(ground_bitstring, "best")
    if n == "rule_bad":
        return rule_layout_for_bitstring(ground_bitstring, "bad")
    if n == "screen_best":
        if ham_file is None:
            raise ValueError("screen_best needs ham_file")
        return EncodingSpec(screen_best_perm(Path(ham_file).name))
    raise ValueError(f"unknown layout {name!r}; choose from {LAYOUTS}")
