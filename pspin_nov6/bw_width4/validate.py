"""Geometry and spectrum checks. No QAOA."""
from __future__ import annotations

from geometry import build_triangles, coloring_report, validate_geometry
from model import cost_diagonal, explicit_cost, gamma_period
from run_round3 import orientation_report


def main():
    g = build_triangles(3, 3)
    assert g["up"][0] == (0, 1, 4)
    assert g["down"][0] == (0, 3, 4)
    assert validate_geometry(g)["ok"]
    assert coloring_report(3, 3, g)["face_coloring_exists"]
    g43 = build_triangles(4, 3)
    g44 = build_triangles(4, 4)
    assert validate_geometry(g43)["ok"] and validate_geometry(g44)["ok"]
    assert not coloring_report(4, 3, g43)["face_coloring_exists"]
    assert not coloring_report(4, 4, g44)["face_coloring_exists"]
    hc = cost_diagonal(g43, 1.0)
    err = abs(hc - explicit_cost(g43, 1.0)).max()
    assert err < 1e-8, err
    per = gamma_period(hc)
    assert abs(per["delta"] - 8.0) < 1e-9
    ori = orientation_report()
    assert ori["same_physics"], ori["full_H_Gamma1_spectrum_max_abs_diff"]
    assert ori["isomorphism_search"]["equivalent_same_orientation"] is not None
    print("validation OK", "E0_4x3", ori["E0_4x3"])


if __name__ == "__main__":
    main()
