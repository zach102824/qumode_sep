# Nov9th: three ideas for robust, fast, shallow fidelity scaling (consultation round 1)

Every idea uses the gate order of `PROTOCOL.md` §2.2 unchanged: \(|+\rangle\to[U_{cZ}\to D(\alpha_\ell)\to S_\ell]^L\), with one shared \(U_{cZ}\), real α, shared cavities, SPSA only, QAOA p = L only, a 1024-query cap, and the seeds and gates of `PROTOCOL.md`. No result below has been produced by this consultation.

## 0. What the local pilot already says (reported, not reproduced)

- At N = 6, 9, 12, Nov3rd and QAOA both clear 0.8 easily. \(F_+\) alone is 0.899 / 0.736 / 0.667 at b = 1, so the absolute thresholds are uninformative there.
- QAOA beats Nov3rd at L = 2, 3 for every reported N. Nov3rd wins only at L = 1 at N = 9 and 12 with b = 1 (N12: 0.9589 vs 0.9169). QAOA reaches F > 0.9 by 64 queries at N = 12, b = 1, while Nov3rd needs about 128 to reach 0.8. The bar is therefore **QAOA at low query caps and N ≥ 12**.
- Shot mode at N = 9, b = 1 with 512 shots per setting is still close to exact for both. Noisy energy selection gives winner's-curse bias up to −0.50, so fresh-shot selection is mandatory (`PROTOCOL.md` §5).
- At N = 15 (8 dev seeds), \(F_+\) = 0.587 at b = 1 and 0.811 at b = 1.5. Nov3rd L = 1 beats QAOA L = 1 in both energy and F on all 8 instances for both fields (b = 1: 0.934 vs 0.889). QAOA wins at L = 2, 3. QAOA is faster at low budget: F at 64 queries is 0.890 vs Nov3rd's 0.692. So Nov3rd's L = 1 edge is a final-quality edge, not a speed edge, and it is the only regime where any non-QAOA circuit currently wins.
- Selection with an 8-candidate shortlist re-ranked by fresh noisy energies plus a sparse monitor (equal 1,048,576 shots per run, N = 12): it beat the original monitor in 36/48 F cells and 38/48 true-energy cells, and moved the median selection bias from −0.315 to −0.013. The exception was Nov3rd b = 1, L = 3. This is development evidence, not universal. It is adopted as the shot-mode selection default (`PROTOCOL.md` §5).
- True-D leakage at α = 0.1 is about 2.4% at L = 1 and about 20.4% at L = 3. This is the single biggest risk for Nov9th.

So every idea is ranked by: (1) paired ΔF against QAOA at caps 128 and 1024 at N = 12 and 15, b = 1; (2) deficit closure η; (3) shot-mode with fresh-shot selection at S = 512.

## Paper (arXiv:2204.02923): source findings vs our hypotheses

Source findings (from the attached PDF):

- **P1.** The GCS ansatz is \(U(y)V(M^{\rm GCS})|\phi(x)\rangle\): free single-spin rotations, then one ZZ entangler \(V=\exp(-\tfrac i4\sum_{n<m}M^{\rm GCS}_{nm}Z_nZ_m)\), then free single-spin rotations. It has only two-body Z terms plus local dressing (Eqs. 3–4).
- **P2.** The fitted \(M^{\rm GCS}\) entries are about Gaussian with zero mean and variance ∝ 1/N, with GOE-like level statistics (⟨r⟩ ≈ 0.53). The authors say the layer parameters "do not necessarily correspond to those of the Hamiltonian".
- **P3.** \(|\phi(x)\rangle\) is Z-polarized in the glass phase and xy-polarized in the paramagnet.
- **P4.** The energy error is worst for 0.5 ≤ g/J ≤ 1.5. The critical field is \(g_C\approx1.5J\). Their \(-\tfrac12\sum_{n,m}J\) equals our \(\sum_{i<j}J\) up to a sign of a symmetric distribution, so our b = 1 and 1.5 sit in the paper's hardest window.
- **P5.** Optimization is natural gradient with adiabatic continuation in g, starting from 10,000 random product states at g = 0. **This is inspiration only. Natural gradient is not allowed here.**
- **P6.** Random weighted graph states \(V(M)|+\rangle\) with GOE \(M\) already reproduce the volume-law S₂, but not the ground-state information carried by \(\phi(x)\) and \(y\).

Our hypotheses (not in the paper):

- **H1.** A shared, J-derived linear combination \(J+\sum\theta_f\hat F_f\) can approximate the instance-specific \(M^{\rm GCS}\) better than \(J\) alone. P2 only says \(M^{\rm GCS}\) is GOE-like, not that it is a function of J.
- **H2.** The real-α displacement, a real Y-type rotation, can partially play the role of \(U(x)\)/\(U(y)\) (P3).
- **H3.** K's sign prior from second-order perturbation theory (Idea 3).
- **H4.** Parameter concentration across instances for normalized features.

The paper has no cavities, no displacement, no SPSA, and nothing about higher Z strings.

---

## Idea 1: N9-LE "leak-echoed, leak-budgeted displacement" (resolve and stress the physical primary)

**Goal.** Make true D optimizable under the anti-exploitation energy \(H^{\rm ext}\) without leakage capping F, and measure how much leakage costs by contrast with the strict-logical model.

**Mechanics.**

- Binary encoding, KM cost (\(n_\theta=2\)), \(H^{\rm ext}\) per-term penalty, Uz/pad-0 extensions (★ defaults).
- Bounded \(\alpha_\ell=0.30\tanh u_\ell\).
- Two leak-control mechanisms that need no new gate:
  - **(a) Echo:** alternate the α signs across layers. With no intervening phases, \(D(\alpha)D(-\alpha)=I\) cancels the first-order 7→8 amplitude. The intervening \(U_{cZ}S\) phases make the cancellation partial, and the logical action survives because the phases differ inside the code.
  - **(b) Level-7 phase:** in binary encoding \(s_3\) acts only on level 7 = |111⟩, the sole source of first-order leakage. SPSA can therefore tune the relative phase between new and returning leak amplitudes. This is specific to binary encoding and absent in Gray.
- At L = 1 neither lever exists. Leakage is ≈ \(C\alpha_1^2\) and must simply be traded off by the energy.

**Initialization (4 starts: E plus 3 independent R draws, matching the local 4-start protocol).**

- Start E: γ = 0.25, θ = 0, \(\alpha_\ell=(-1)^{\ell+1}\cdot0.06\), s = 0.
- Start R: random box with γ ∈ U(−0.5, 0.5), θ ∈ U(−0.5, 0.5), α ∈ U(−0.08, 0.08), s ∈ U(−0.3, 0.3). Stream 2 RNG.

**SPSA.** `PROTOCOL.md` §5 with σ_α = 0.02. Calibrate c on α separately, because the leak term makes the landscape stiff in α: require \(c\sigma_\alpha\) to change \(p_{\rm leak}\) by ≤ 0.01 at x0.

**Arms (same seeds, same budget).**

1. Nov9th-leaky (Uz)
2. Nov9th-leaky (U0)
3. Nov9th-B strict-logical with \(D_8\) (diagnostic)
4. Nov9th-leaky with α signs locked equal (no echo)
5. Nov9th-leaky with \(s_3\equiv0\)
6. H0 negative control (selection forbidden; demonstrates exploitation)
7. Nov3rd
8. QAOA p = L

**Physical tests.** T4, T5, T7, T8, T12 plus the NF ladder at selected points, \(p_{\rm leak}\) per layer, f ≥ 16 aliasing, and ⟨Π⟩.

**Falsification (dev N = 9 and 12, b = 1, exact).**

- K1: leaky median η < 0.25 at every L while B median η ≥ 0.5. Then leakage, not expressivity, kills Nov9th. Stop leaky scaling claims and escalate decision D6/D7 to the user.
- K2: B also has η < 0.25. Then the architecture itself (real-α Y-mixer, parity breaking) is the limit. Hand over to Ideas 2 and 3, or recommend complex α.
- K3: the echo arm is no better than locked signs (median \(p_{\rm leak}\) difference < 0.01 at L = 3). Then mechanism (a) is dead.
- K4: \(s_3\equiv0\) is no worse. Then mechanism (b) is dead.
- Uz vs U0 median |ΔF| > 0.01 means the result is extension-dependent; flag it.

**Prediction to check (not a result).** L = 1 is weak (\(O(\gamma^2)\), `PROTOCOL.md` §9). The best leaky L is 2 or 3, with \(p_{\rm leak}\) ≤ 3%.

## Idea 2: N9-GR "encoding route co-design" (which logical spin rides which Fock edge)

**Goal.** In binary encoding, D's logical action is very unequal across bits.

- Bit 0 is flipped on all 7 code edges.
- Bit 1 only on 1–2 and 5–6, jointly with bit 0.
- Bit 2 only on 3–4, jointly with bits 0 and 1.

Assigning spins to cavities and bit roles is therefore a free, charged classical choice that shapes the mixer. It adds no gate and no continuous control.

**Mechanics.** Three preregistered J-only rules, each \(O(N^3)\), computed once per instance:

- **G (grouping):** greedy triples maximizing intra-cavity \(\sum|J|\). Rationale: joint bit-0/bit-1 flips preserve \(z_az_b\), so the cheap correlated fluctuations of a strongly coupled pair are native to the 1–2 and 5–6 edges.
- **R (roles):** within each cavity, the pair with the largest |J| takes bits 0 and 1. Of that pair, the spin with the smaller row norm \(\|J_{i\cdot}\|_2\) takes bit 0, because the weakest-pinned spin is expected to fluctuate most. The third spin takes bit 2.
- **V (variant encoding):** Gray route ("Nov9th-Gray", labeled historical), where every code edge is a single-bit flip. Bit 0 gets 4 edges, bit 1 gets 2, bit 2 gets 1, and \(s_3\) loses its level-7 isolation.

**M feature.** \(\hat M\) is recomputed on the chosen grouping. K is grouping-independent. Report Pearson corr(\(\hat M\) diag, \(H_{ZZ}\) diag) per grouping, as a classical-only diagnostic.

**Initialization / SPSA.** Same two starts and schedule as Idea 1, with the Idea 1 echo init kept so the ideas differ only in encoding.

**Arms.**

1. interleaved + identity roles (Nov 8 default)
2. G + identity
3. G + R
4. interleaved + R
5. G + anti-R (reverse role order, the falsifier)
6. 3 random groupings × random roles (seeded, stream 2), to measure the variance an encoding choice induces
7. Gray + G + R
8. Nov3rd
9. QAOA

**Physical tests.** As in Idea 1. Also T9 for each encoding, and leak by cavity role.

**Falsification.**

- G + R does not beat the median of the random-encoding arms by median ΔF ≥ 0.02 at N = 12, b = 1, with paired-bootstrap CI > 0: encoding design is not a lever.
- anti-R ≥ R: the role rationale is wrong.
- If the random-encoding spread (IQR of per-instance F across the 3 random encodings) exceeds 0.05, encoding is a hidden degree of freedom. Every Nov9th result must then state its encoding, and the default must be justified.

## Idea 3: N9-FT "GCS-inspired features + concentrated transfer" (cost side and query scaling)

**Goal.**

- (i) Give the single shared entangler a better-shaped coupling matrix (H1).
- (ii) Turn the per-instance search into a short finetune from instance-independent parameters (H4), so queries per instance stay flat as N grows. This is the "fast" part QAOA already has at N = 12.

**Feature library** (all RMS-matched with `hat`, all shared θ, \(n_\theta\) counted):

- K (squared local fields).
- M (block mean, interleaved grouping).
- **R1** = offdiag\((\lambda_{\min}v_{\min}v_{\min}^\top)\), the most negative eigenpair of J. Hypothesis: low classical energy aligns with the bottom spectral direction, and R1 upweights phase along it.
- **Optional R_r** with \(r=\lceil N/6\rceil\) bottom eigenpairs.

**K sign prior (H3).** Second-order perturbation theory in b gives \(E^{(2)}=-b^2\sum_i 1/(2|h_i|)\). Expanding in \(h_i^2\) around \(\bar h^2=\sum_jJ_{ij}^2\approx1\) gives \(+\tfrac{b^2}{2\bar h^3}\sum_{j<k}K_{jk}z_jz_k\), so the raw-units init is \(\theta_K^{\rm raw}\approx b^2/2\), which in hat units is \(\theta_K=\theta_K^{\rm raw}\,\mathrm{RMS}(K)/\mathrm{RMS}(J)\). b ≥ 1 is outside the perturbative regime, so this sets only an init and a sign, and is ablated against θ = 0.

**Higher Z strings.** The next order gives \(\sum_ih_i^4\), which is dense 4-body with \(O(N^4)\) terms. Its compiled cost is unapproved, so it is **not** in round 1 and is listed as D8. No ZZZ: it is Z2-odd and would add to Nov9th's parity problem.

**Transfer (SPSA only).**

- Phase T-train, per (N_train = 9, b, L): one SPSA run on the mean energy over a minibatch of 4 dev instances that changes each iteration. These are dev seeds 10000..10015 only, never validation or final. Cap 4096 queries, each minibatch energy counted as 4.
- The resulting shared parameter vector is \(x^\star\).
- Phase T-use, per instance at N = 9, 12, 15: (a) zero-shot \(x^\star\) (2 queries: evaluation + fresh-shot check); (b) finetune from \(x^\star\) with 1 start × 256 queries; (c) cold 1024 queries as control.
- QAOA gets the identical pipeline: same minibatches, same caps.
- Amortized cost is reported as (T-train queries)/(#instances served) + per-instance queries.

**Arms (2 × 2 factorial + controls).** {KM, KM + R1} × {cold-1024, transfer + 256}, plus KM + K-prior init, plus QAOA {cold, transfer}, plus Nov3rd cold.

**Physical tests.** Same as Idea 1, plus transfer of NF_sim (cutoff re-check at \(x^\star\) for N = 15).

**Falsification.**

- R1 adds < 0.01 median F at N = 12 and the K-prior is no better than θ = 0. Then H1/H3 are unsupported, and the KM baseline stays.
- The transfer + 256 median F is below cold-1024 by more than 0.03 at N = 15, or the QAOA transfer pipeline wins the paired ΔF at cap 256. Then the "fast" claim is not supported for Nov9th.
- \(|\theta_f|\) pinned at bounds: features are mis-scaled. Report it; do not silently re-scale.

---

## Assignment for three cloud workers (independent, same base)

- Every worker implements `PROTOCOL.md` themselves and passes T1–T14 before any dev table. All use binary, Uz, pad-0, and per-term \(H^{\rm ext}\) defaults, labeled provisional until DECISIONS D1–D5 are confirmed.
- Worker 1 runs Idea 1, which includes the strict-logical contrast every other idea relies on for interpretation. Worker 2 runs Idea 2. Worker 3 runs Idea 3.
- Order inside each worker:
  1. S0
  2. S1 at N = 6 (8 dev seeds) as a smoke test
  3. S1 at N = 9, 12 (32 dev seeds)
  4. S2 at N = 15 (32 dev seeds) plus budget curves
  5. S3 shots
- Report against the common ranking (§0). Nobody touches validation or final.
- The local 8-core stream continues Nov3rd/QAOA at N = 12. It should add the \(F_+\), η, \(\Delta_{\rm odd}\), and budget-curve columns so cloud and local tables join on (seed, N, b, L, method).

Rough compute (exact mode): N = 12 with NF_sim 16–24 gives 65k–330k amplitudes, so a 1024-query run takes seconds. N = 15 gives 1M–8M amplitudes, about minutes per run. N = 18 with NF 16 gives 16.7M amplitudes (268 MB complex128). The N = 18 final for 1000 seeds × 2 b × 3 L × 3+ methods needs GPUs or a reduced method set, decided at FREEZE.
