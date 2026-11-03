# QAOA L=1 — paper-init n6 train vs Nov_3rd warm/polished

_Generated 2026-11-03 15:50:40 +0800 (Asia/Shanghai)_

- **QAOA_n6:** paper init β=0.01 γ=-0.01, nstart=1 → trained β=[0.7366378307342529], γ=[-5.233087062835693], loss=2.2671
- **QAOA_n912:** prior n9+n12 multi-start β=[0.7718809843063354], γ=[-5.4505720138549805]
- **Nov warm:** QAOA shell from OLD n9+n12 angles (exact TIE to old QAOA)
- **Nov polished:** free LBFGS on n6×1000

## scale1000 B=1000

| n | QAOA_n6 Med | QAOA_n6 B90 | QAOA_n912 Med | warm Med | polished Med | R_warm/n6 | gate_w | R_pol/n6 | gate_p | R_n912/n6 |
|---|-------------|-------------|---------------|----------|--------------|-----------|--------|----------|--------|-----------|
| 6 | 9.810 | 18.704 | — | 9.779 | 9.502 | 0.997× | TIE | 0.969× | BEAT | — |
| 9 | 29.309 | 62.636 | 29.279 | 29.279 | 29.537 | 0.999× | TIE | 1.008× | TIE | 0.999× |
| 12 | 91.042 | 214.158 | 90.033 | 90.033 | 93.283 | 0.989× | TIE | 1.025× | UNDER | 0.989× |
| 15 | 293.232 | 747.637 | 289.892 | 289.893 | 318.811 | 0.989× | TIE | 1.087× | UNDER | 0.989× |

## Notes

- Gate: BEAT if R<0.98, TIE if 0.98≤R≤1.02, UNDER if R>1.02.
- Denominator is **n6-trained paper-init QAOA** (fair train budget vs Nov_3rd).
- Angle convention: qaoa_ref U_C=exp(-iγ Cn), U_B=exp(-iβ X/2).
- Init: arXiv:2208.06909 §Parameter optimization.
