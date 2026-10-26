#!/usr/bin/env python3
"""Build a padded 5-island checkpoint that includes disk best/ into the live population.

Copies all programs from the latest checkpoint_*, injects best/best_program.py +
best_program_info.json as a full Program JSON, pads island metadata to 5 islands,
sets best_program_id to the disk winner, and writes checkpoint_<N> where N is
max(last_iteration_preferred, latest+1) so driver latest() picks it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RUN = Path("runs/gnn_tiecost")
WINNER_ID = "21df3d9d-63e1-450d-bc8c-5d243b4dc51b"
NUM_ISLANDS = 5
# Feature cell from evolve.log when winner was admitted (island 1).
WINNER_FEATURE_KEY = "8-4"
WINNER_ISLAND = 1


def latest_ckpt(ck_root: Path) -> Path:
    cks = sorted(ck_root.glob("checkpoint_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
    if not cks:
        raise SystemExit("no checkpoints found")
    return cks[-1]


def main() -> None:
    root = Path(__file__).resolve().parent
    run = root / RUN
    ck_root = run / "checkpoints"
    src = latest_ckpt(ck_root)
    src_n = int(src.name.rsplit("_", 1)[-1])

    best_info = json.loads((run / "best" / "best_program_info.json").read_text())
    best_code = (run / "best" / "best_program.py").read_text()
    if best_info.get("id") != WINNER_ID:
        print(f"WARN: disk best id {best_info.get('id')} != expected {WINNER_ID}", file=sys.stderr)

    meta = json.loads((src / "metadata.json").read_text())
    # Template: parent if present, else any program.
    parent_id = best_info.get("parent_id")
    prog_dir = src / "programs"
    template_path = prog_dir / f"{parent_id}.json"
    if not template_path.exists():
        template_path = next(prog_dir.glob("*.json"))
    template = json.loads(template_path.read_text())

    winner = {
        "id": WINNER_ID,
        "code": best_code,
        "changes_description": template.get("changes_description")
        or (template.get("metadata") or {}).get("changes")
        or "reinjected disk best",
        "language": best_info.get("language", "python"),
        "parent_id": parent_id,
        "generation": int(best_info.get("generation", 2)),
        "timestamp": float(best_info.get("timestamp", template.get("timestamp", 0.0))),
        "iteration_found": int(best_info.get("iteration", 8)),
        "metrics": best_info.get("metrics", {}),
        "complexity": float(template.get("complexity") or 0.0),
        "diversity": float(template.get("diversity") or 0.0),
        "metadata": {
            "changes": "reinjected from runs/gnn_tiecost/best after 5-island bump",
            "parent_metrics": (template.get("metadata") or {}).get("parent_metrics"),
            "island": WINNER_ISLAND,
            "token_usage": {},
            "llm_model": "reinject",
            "reinjected": True,
        },
        "prompts": None,
        "artifacts_json": best_info.get("metrics", {}).get("_artifact_json"),
        "artifact_dir": None,
        "embedding": None,
    }

    # Pad islands / feature maps / bests / gens to NUM_ISLANDS
    islands = [list(x) for x in meta.get("islands", [])]
    feature_maps = [dict(x) for x in meta.get("island_feature_maps", [])]
    island_bests = list(meta.get("island_best_programs", []))
    island_gens = list(meta.get("island_generations", []))

    while len(islands) < NUM_ISLANDS:
        islands.append([])
    while len(feature_maps) < NUM_ISLANDS:
        feature_maps.append({})
    while len(island_bests) < NUM_ISLANDS:
        island_bests.append(None)
    while len(island_gens) < NUM_ISLANDS:
        island_gens.append(0)

    # Trim if somehow longer (shouldn't be)
    islands = islands[:NUM_ISLANDS]
    feature_maps = feature_maps[:NUM_ISLANDS]
    island_bests = island_bests[:NUM_ISLANDS]
    island_gens = island_gens[:NUM_ISLANDS]

    # Place winner on WINNER_ISLAND
    if WINNER_ID not in islands[WINNER_ISLAND]:
        islands[WINNER_ISLAND].append(WINNER_ID)
    feature_maps[WINNER_ISLAND][WINNER_FEATURE_KEY] = WINNER_ID
    island_bests[WINNER_ISLAND] = WINNER_ID
    island_gens[WINNER_ISLAND] = max(island_gens[WINNER_ISLAND], winner["generation"])

    archive = list(meta.get("archive", []))
    if WINNER_ID not in archive:
        archive.append(WINNER_ID)

    last_iter = max(int(meta.get("last_iteration", src_n)), int(winner["iteration_found"]), 8)
    out_n = max(src_n + 1, last_iter)  # prefer checkpoint_8 when src is 5
    out = ck_root / f"checkpoint_{out_n}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "programs").mkdir()

    # Copy all existing programs
    for p in prog_dir.glob("*.json"):
        shutil.copy2(p, out / "programs" / p.name)
    # Write winner
    (out / "programs" / f"{WINNER_ID}.json").write_text(json.dumps(winner, indent=2) + "\n")

    new_meta = {
        "island_feature_maps": feature_maps,
        "islands": islands,
        "archive": archive,
        "best_program_id": WINNER_ID,
        "island_best_programs": island_bests,
        "last_iteration": last_iter,
        "current_island": int(meta.get("current_island", 0)) % NUM_ISLANDS,
        "island_generations": island_gens,
        "last_migration_generation": int(meta.get("last_migration_generation", 0)),
        "feature_stats": meta.get("feature_stats", {}),
    }
    (out / "metadata.json").write_text(json.dumps(new_meta, indent=2) + "\n")

    # Mirror OE checkpoint best_* files
    (out / "best_program.py").write_text(best_code)
    ck_best_info = {
        "id": WINNER_ID,
        "generation": winner["generation"],
        "iteration": winner["iteration_found"],
        "current_iteration": last_iter,
        "metrics": winner["metrics"],
        "language": winner["language"],
        "timestamp": winner["timestamp"],
        "saved_at": best_info.get("saved_at"),
    }
    (out / "best_program_info.json").write_text(json.dumps(ck_best_info, indent=2) + "\n")

    m = winner["metrics"]
    print(f"wrote {out}")
    print(f"  from {src}")
    print(f"  best_program_id={WINNER_ID}")
    print(f"  score={m.get('combined_score')} n6_med={m.get('n6_med')} n8_med={m.get('n8_med')}")
    print(f"  last_iteration={last_iter} islands={[len(i) for i in islands]}")
    print(f"  programs={len(list((out/'programs').glob('*.json')))}")


if __name__ == "__main__":
    main()
