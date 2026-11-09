# Results

The audit is in `AUDIT.md`. Numerical residuals are in `validation.json`.

Same frozen N=12 target, stored p=12 energy-selected state, recomputed and not reoptimized:

- `E = -8.325407549783135`
- full-space `F = |⟨GS|ψ⟩|^2 = 0.5147185362888936`
- `Var(H) = 0.42114525455677665`
- post-hoc `η = (E − E_max)/(E_0 − E_max) = 0.9911210589175745`
- `E_0 = -8.613202458935264`, `E_max = 23.79999934975569`

Matched status against arXiv:1906.02700: incomparable. The paper's printed performances are η at other `(N, p, J, B)`. No N=12 p=12 squared fidelity is published. Dense N=6 agreement and the gradient check passed. No bug, no superseded results, no new target.
