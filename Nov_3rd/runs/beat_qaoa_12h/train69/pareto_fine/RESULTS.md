# pareto_fine — L=2 α grid around the n15 knee

Train {6,9} only. Loss = n9 batch median log-ratio vs QAOA. CORE eval {6,9,12,15}, B=1000. No n12/n15 in the loss.

- α ∈ {0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85}
- 6 polish seeds, Adam 400, lr 3e-4, δ ball 0.025 on (c2, c3)
- Endpoints: n9med ρ0.05 seeds 7, 5, 2 × ratio69 s0 and s1 (s7×s1 is the box α=0.80 pair; s7×s0 not run)
- Raw convex blend scored before polish
- **ALL_CORE_BEAT = false** (280 evals)

## Best n15

| n15 | n6 | n9 | n12 | what |
|-----|----|----|-----|------|
| **0.9850** | 0.9598 BEAT | 0.9822 TIE | 0.9817 TIE | polish t5×s0 α=0.70 s3 (ball init) |
| 0.9856 | 0.9596 BEAT | 0.9831 TIE | 0.9824 TIE | raw t7×s1 α=0.65 |
| 0.9867 | 0.9583 BEAT | 0.9797 BEAT | 0.9856 TIE | polish t5×s0 α=0.72 s4 (n6+n9 BEAT) |
| 0.9891 | 0.9601 BEAT | 0.9800 BEAT | 0.9780 BEAT | polish t5×s1 α=0.70 s3 (n6+n9+n12 BEAT) |

Box reference: pareto_blend α=0.80 s1, n15 0.9853, n9 0.9782 BEAT, n12 0.9858 TIE.

n9-only Adam improves n9/n12 and gives up n15. The 0.9850 point is the minimax of {n6,n9,n12,n15}; the worst coordinate is still n15 at 0.9850, about 0.005 above 0.98.
