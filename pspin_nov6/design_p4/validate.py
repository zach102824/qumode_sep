"""Fast checks for the SQS constructions and the cost diagonal. No QAOA."""
from __future__ import annotations

import numpy as np

from ham import explicit_hc_from_pauli, gamma_period_from_diag, hc_diag_for_pattern, signs_pm_from_eps
from signs import prepare_catalog
from sqs import code_bitmasks, design_blocks, verify_sqs


def main():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    for n in (8, 10, 14, 16):
        blocks = list(design_blocks(n))
        verify_sqs(n, blocks)
        rank, code, b = code_bitmasks(n, blocks)
        all1 = (1 << n) - 1
        check(int(code[0].sum()) == 0, f"B0 N={n}")
        check(int(code[all1].sum()) == 0, f"all-ones not in ker B, N={n}")
        check(rank == int(np.round(np.log2(len({tuple(row.tolist()) for row in code})))), f"rank vs image N={n}")

    cat = prepare_catalog(8, compute_aut=True)
    check(cat["aut_exhaustive"] and cat["aut_order"] == 1344, f"Aut SQS(8) {cat['aut_order']}")
    cover = cat["covering_radius"]
    check(cover and cover["certified"] and cover["R"] == 6 and cover["n_cosets_at_R"] == 1, f"cover {cover}")
    # point star gauges to the unfrustrated coset
    star = next(c for c in cat["catalog"] if c["name"] == "point_star_0")
    check(star["u_min"] == 0 and star["n_neg_blocks"] == 0, "point-star is gauge-equivalent to uniform on N=8")
    aff = next(c for c in cat["catalog"] if c["name"] == "affine_off_origin")
    check(aff["u_min"] == 6, f"affine frustration {aff['u_min']}")
    for name in ("uniform", "block_0", "affine_off_origin"):
        c = next(x for x in cat["catalog"] if x["name"] == name)
        diag = hc_diag_for_pattern(8, cat["blocks"], c["eps"])
        ref = explicit_hc_from_pauli(8, cat["blocks"], signs_pm_from_eps(c["eps"], cat["b"]))
        err = float(np.max(np.abs(diag - ref)))
        check(err < 1e-8, f"diag {name} {err}")
        per = gamma_period_from_diag(diag)
        # phases close after one period
        ph = np.exp(-1j * per["T_gamma"] * diag)
        ph = ph / ph[0]
        check(float(np.max(np.abs(ph - 1))) < 1e-8, f"period {name}")

    cat10 = prepare_catalog(10, compute_aut=True)
    check(cat10["aut_exhaustive"] and cat10["aut_order"] == 1440, f"Aut SQS(10) {cat10['aut_order']}")
    uni = next(c for c in cat10["catalog"] if c["name"] == "uniform")
    check(uni["degeneracy"] == 2 and uni["u_min"] == 0, "N=10 uniform is a 2-point classical manifold")

    print({"n_failures": len(failures), "failures": failures})
    if failures:
        raise SystemExit(1)
    print("validation OK")


if __name__ == "__main__":
    main()
