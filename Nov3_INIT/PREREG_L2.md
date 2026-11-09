# N3INIT L2 evaluation

Not a new arm search. v2 already adopted `center4` and v3 rejected `spread8`. This run only asks whether that frozen comparison at `L=2` looks like `L=1` or like `L=3`.

- `N ∈ {9,12}`, `L=2`, `q ∈ {128,512,1024}`, `Q = q*2` so `256`, `1024`, `2048`.
- `equal4` for Nov3 (method id 1) and QAOA (method id 0), same boxes and gains as v1.
- `center4` for Nov3 only, method id 1, start 0 coherent, starts 1–3 identical to `equal4`. QAOA is the `equal4` run, not a second copy.
- Seeds `10018..10025`. No adoption rule. `center4` stays the recommended init. These numbers do not retune it.
- Same energy-only SPSA, same `B*` files, validation and final seeds refused.
