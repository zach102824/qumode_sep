# scale1000 regen (SCB=54e9) — CORE n∈{6,9,12,15}

`Nov_3rd/data/scale1000/` was missing on this box (npz gitignored). Regenerated with
`regen_scale1000.py`: `sat_gen.instances(n, 1000, 54_000_000_000, compute_E=True)`, stored as
`scale1000_*` and identical `test_*` keys (E uint8).

## Content check — PASS (bit-exact)

QAOA L=2 (`qaoa_angles/L2/angles.json`) on the regenerated packs reproduces
`qaoa_angles/L2/results.json` Med and B90 to all printed digits for n=6,9,12,15
(`qaoa_L2_ref_check.log`). L=16 `ratio69_w9` seed0 also re-evals to the
committed n6/n9 Med (1.1568 / 1.4069).

## File sha vs DATA_MANIFEST — not byte-identical

| n | manifest bytes | regen bytes | Δ | manifest sha | regen sha |
|---|----------------|-------------|---|--------------|-----------|
| 6 | 235607 | 233320 | −2287 | d2e17f2c5e70 | fe50ee80e43a |
| 9 | 814231 | 811938 | −2293 | 49def928b148 | 8454dc560264 |
| 12 | 4343829 | 4341530 | −2299 | 4c85d564b784 | 3de9c44d2f8e |
| 15 | 31237905 | 31235604 | −2301 | c5279c0aa15d | 69a467b170fb |

Δ is ~constant across n ⇒ extra small zip entries in the original (empty `train_`/`val_`
placeholders explain 1926 B; ~360 B ≈ two more scalar keys of unknown name), not different data.
The original builder script is not in the repo. Array-content sha256 in `regen_hashes.json`.

n18 not regenerated (not CORE).
