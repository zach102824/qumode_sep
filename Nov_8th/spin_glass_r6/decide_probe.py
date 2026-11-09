"""Apply the frozen energy gate. Does not read fidelity."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "PREREG.json").read_text())
GAIN = float(PREREG["probe"]["energy_gain"])


def main():
    raw = json.loads((HERE / "probe_raw.json").read_text())
    gate = [c for c in raw["cells"] if c["in_gate"]]
    if len(gate) != 8:
        raise SystemExit(f"expected 8 gate cells, got {len(gate)}")
    improved = [c for c in gate if c["dE"] <= -GAIN]
    verdict = {
        "n_gate": len(gate),
        "n_improved": len(improved),
        "gain": GAIN,
        "launch_confirmation": len(improved) >= 4,
        "improved": [{"N": c["N"], "k": c["k"], "dE": c["dE"], "label": c["best_label"]} for c in improved],
        "all_gate_dE": [{"N": c["N"], "k": c["k"], "dE": c["dE"], "label": c["best_label"]} for c in gate],
    }
    (HERE / "probe_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: verdict[k] for k in ("n_improved", "launch_confirmation")}))


if __name__ == "__main__":
    main()
