# B* harness

Zach at 19:05 Asia/Shanghai on 2026-11-09: the next comparison focuses on the maximum-entropy field. The finished b = 1 and b = 1.5 pilot stays the reference. This harness does not search for B* and does not start SPSA.

## Manifest

```json
{
  "instances": [
    {"seed": 10000, "N": 6, "B_star": 1.0, "source": "filled_by_the_next_assignment"}
  ]
}
```

`B_star` is the positive coefficient in `H = sum J_ij Z_i Z_j - B_star sum X_i`. One value per `(seed, N)`. Seeds 11000–11031 and 20000–20999 are rejected.

## Fixed cut

Side A is sites `0 .. N//2 - 1` (the least significant bits). Side B is the rest. For odd N the sides differ by one site. The cut is not varied to increase the entropy.

`cut_entropy` is the von Neumann entropy of that cut, in nats. `gs_contract` checks a strictly positive even ground state, a positive gap, a positive odd gap, and a ground-state residual at most `1e-8`.

## Circuits prepared, not run

The job list is Nov3rd, interleaved true-displacement Nov9th (`inter_id`, Uz, binary), and QAOA at p = L. The GR encoding ablations are not in this list. The SPSA stream key remains `round(10 * B_star)`.

A cutoff certificate from b = 1 and b = 1.5 is not applied to a new B*. The displacement state does not depend on the field, and the certified energy does.

```bash
python3 bstar_harness.py self-check
python3 bstar_harness.py dry-jobs manifest.json
```

`launch` raises. There is no batch runner.
