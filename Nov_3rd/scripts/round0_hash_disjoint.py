#!/usr/bin/env python3
"""Hash-check sealed train seeds vs scale1000 eval seeds (must be disjoint)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd import settings as S

OUT = S.OUT / "beat_qaoa_12h" / "round_0"


def seed_set(pack_path: Path, prefer):
    d = np.load(pack_path, allow_pickle=True)
    for k in prefer:
        if k in d.files:
            arr = np.asarray(d[k]).ravel()
            if arr.size == 0:
                return set(), k, 0
            return set(map(int, arr.tolist())), k, int(arr.size)
    return None, None, 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    ok = True
    for n in [6, 9, 12, 15, 18]:
        sealed = S.SEALED_DATA / f"data_n{n}.npz"
        scale = S.SCALE1000_DIR / f"data_n{n}.npz"
        if not sealed.exists() or not scale.exists():
            rows[n] = {"status": "missing_pack"}
            continue
        sa, ska, na = seed_set(sealed, ["train_seeds", "seeds"])
        sb, skb, nb = seed_set(scale, ["scale1000_seeds", "test_seeds", "seeds"])
        if sa is None or sb is None:
            rows[n] = {"status": "no_seeds", "sealed_key": ska, "scale_key": skb}
            ok = False
            continue
        inter = sorted(sa & sb)
        # also hash train_E / scale1000_E first 16 bytes fingerprint
        def fp(path, key):
            d = np.load(path, allow_pickle=True)
            if key not in d.files:
                return None
            a = np.asarray(d[key])
            h = hashlib.sha256(a.tobytes() if a.dtype != object else str(a.tolist()).encode())
            return {"key": key, "shape": list(a.shape), "sha256_16": h.hexdigest()[:16]}

        rec = {
            "status": "ok",
            "sealed_train_n": na,
            "sealed_key": ska,
            "scale_n": nb,
            "scale_key": skb,
            "intersect": len(inter),
            "disjoint": len(inter) == 0,
            "sealed_E": fp(sealed, "train_E"),
            "scale_E": fp(scale, "scale1000_E"),
        }
        if len(inter) != 0:
            ok = False
            rec["sample_inter"] = inter[:10]
        if na == 0 and n in (15,):
            rec["note"] = "sealed train empty — Round1 must generate sealed train pack"
        rows[n] = rec

    payload = {
        "pass": ok and all(rows.get(n, {}).get("disjoint") for n in [6, 9, 12]),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "tz": "Asia/Shanghai",
    }
    path = OUT / "hash_disjoint.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("wrote", path, "pass=", payload["pass"])
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
