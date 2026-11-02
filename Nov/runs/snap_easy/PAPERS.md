# SNAP easiness — papers & implementability notes

_Asia/Shanghai 2026-11-02. Goal: restrict bare SNAP so hardware/control is easiest while Nov1st-final Motif stays locked._

Native SNAP = \(S(\vec\theta)=\sum_n e^{i\theta_n}|n\rangle\langle n|\). Hardware cost scales with **how wild** \(\theta_n\) is (number of selective sidebands / pulse comb tones, spectral resolution vs \(\chi\)).

## Key references

1. **Heeres et al., PRL 115, 137002 (2015)** — [arXiv:1503.01496](https://arxiv.org/abs/1503.01496)
   Introduces SNAP: geometric phase via two selective qubit rotations at \(\omega_q+n\chi\). Notes that **linear** \(\theta_n=n\phi\) is a cavity IQ rotation and **quadratic** \(\theta_n\propto n^2\) is **Kerr evolution** — both are “simple phase relations,” while arbitrary \(\theta_n\) needs a full frequency comb. Also uses SNAP to **cancel Kerr**.
   → Restriction: prefer Kerr / low-degree polynomial \(\theta(n)\); few nonzero \(\theta_n\) = fewer tones.

2. **Krastanov et al., PRA 92, 040303(R) (2015)** — [universal SNAP+displacement](https://jianggroup.yale.edu/sites/default/files/physreva.92.universal_control.pdf)
   Proves SNAP + displacements are universal. Explicitly: choosing \(\theta_n\propto n^2\) **simulates Kerr**; SNAP can emulate any number-conserving nonlinearity.
   → Restriction: a **single-parameter Kerr SNAP** is the cheapest nonlinear SNAP that still adds non-Gaussianity beyond displacements.

3. **Fösel / Kjaergaard / Marquardt et al., arXiv:2004.14256** — Efficient cavity control with SNAP gates
   Hierarchical insertion + co-optimization: many targets need only **3–4 SNAP** pulses (vs ~50 naive). Modular SNAP beats GRAPE on interpretability; SNAP can be made error-transparent.
   → Restriction: shorter SNAP sequences matter as much as simpler \(\theta_n\); shared/repeated SNAP params across layers help.

4. **Landgraf et al., arXiv:2310.10498** — Fast cavity control / improved SNAP pulses
   Keeps the **Gaussian selective-pulse** SNAP shape; suppresses coherent errors without adding frequencies. Hardware still: dispersive transmon + narrowband Gaussian comb.
   → Restriction: “Gaussian gate” in the lab sense = **Gaussian envelope selective drives**, not a Gaussian phase law. Our `gauss` head = **quadratic phase on popcount** (analytic Kerr-like), which maps to a smooth tone amplitude schedule.

5. **Floquet-engineered fast SNAP, arXiv:2506.03034**
   Speeds SNAP in weak-\(\chi\) cavities via Floquet sidebands; example gate \(e^{i\pi|0\rangle\langle0|}\) = **single-level (proj-like) SNAP**.
   → Restriction: **few_m / proj** (only lowest Fock phases free) is the experimentally cheapest selective SNAP.

6. **Polynomial / cubic phase via Rabi (arXiv:2609.09132 and related CV work)**
   Degree-\(R\) phase gates \(e^{iT p(\hat X)}\) (cubic = \(x^3\)) are the “easy” non-Gaussian CV primitives; SNAP can **exactly** realize polynomial \(\theta_n\) on a truncated Fock space by setting \(\theta_n=p(n)\).
   → Restriction: `poly2_n` / `poly3_n` / `kerr` are the SNAP forms that match native oscillator Hamiltonians (Kerr, higher-order self-Kerr) without a free \(\theta_{0..7}\) table.

## Restriction → implementability (our ladder)

| Mode | #params/L | Why “easy” on hardware |
|------|-----------|-------------------------|
| `const` | 1 | Global phase only — not useful (sanity kill) |
| `kerr` | 2 | Idle Kerr + frame: \(\theta_k=\chi k(k-1)/2+\omega k\) — no selective comb, or one calibrated nonlinearity |
| `gauss` | 3 | Quadratic on **popcount** (BinH shells) — smooth \(\theta\), 3 tones’ worth of freedom |
| `poly2_n` | 3 | Quadratic on Fock \(n\) — pure Kerr-family |
| `few_m` (M=2,3) | M | Only drive M selective sidebands; rest identity |
| `gauss_plus1` | 4 | Gauss + one shell residual — almost shell, one extra tone |
| `poly3_*` | 4 | Cubic phase / cubic Kerr |
| `shell` | 4 | Current sealed reference (popcount table) |
| `free` | 8 | Full SNAP table — hardest (8 independent phases) |

**Shared-across-L** variants cut total SNAP params by \(\times L\) (one pulse recipe reused).

## Takeaway for this repo

Bare SNAP is **easiest** when \(\theta_n\) is (i) low-degree polynomial in \(n\) or \(pc(n)\), (ii) sparse (few_m / proj), or (iii) shared across layers. Full free/shell tables need a full selective frequency comb every layer. Explore tracks which of these still stays near sealed Med/B90 under locked Nov1st-final.
