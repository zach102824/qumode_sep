# Fourier-γ deepen — projector-gadget (unsat) BinH

_Generated 2026-10-30 19:27:26 +0800 (Asia/Shanghai)_

q=3 L_train=16 fourier_phi=True. Scale1000 aborted early after L16 proved worse; L20 ladder also fails gate vs free-γ projector.

**keep=False — DISCARD**

## Baseline projector scale1000
| n | Med | B90 |
|---|-----|-----|
| 9 | 1.445 | — |
| 12 | 1.926 | — |
| 15 | 2.820 | — |

## Deploy ladders

### L=16

#### val
| n | Med | B90 |
|---|-----|-----|
| 9 | 1.889 | 3.414 |
| 12 | 2.904 | 8.683 |
| 15 | 3.118 | 7.773 |
| 18 | 7.315 | 44.440 |

#### test
| n | Med | B90 |
|---|-----|-----|
| 9 | 1.764 | 2.641 |
| 12 | 2.955 | 12.542 |
| 15 | 5.860 | 24.287 |
| 18 | 4.585 | 13.597 |

### L=20

#### val
| n | Med | B90 |
|---|-----|-----|
| 9 | 1.563 | 2.782 |
| 12 | 2.365 | 8.873 |
| 15 | 2.543 | 7.041 |
| 18 | 6.930 | 26.856 |

#### test
| n | Med | B90 |
|---|-----|-----|
| 9 | 1.701 | 2.361 |
| 12 | 2.650 | 7.598 |
| 15 | 5.102 | 18.228 |
| 18 | 3.937 | 14.986 |

## Partial scale1000 (L=16 only, aborted)
| n | Med | B90 | vs base Med |
|---|-----|-----|-------------|
| 9 | 1.843 | 4.166 | ×0.784 |

## Verdict

- L16 scale n9 Med=1.843 > baseline 1.445 (worse).
- L20 test n15 Med=5.102 > projector test 4.167; val n15=2.543 only mild vs free-γ.
- **Do not replace** `unsat_params.pt`. Stick with free γ/φ projector-gadget.
- Scale abort intentional to unblock n18 B=1000 eval.

