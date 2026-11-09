# STARTS S8 — equal-budget eight starts is not adopted

Continuation is closed. This batch asked whether L=3 fade at N15 is a start-diversity failure under a matched query cap. Ordinary C0 SPSA, same sigma and per-start calibration/gain as historical C0. S4 is four equal starts (TQA + 3 U(−0.6,0.6), K=379). S8 is eight equal starts (same streams for 0–3 plus four more, K=187). Q=3072 inclusive for every arm. Seeds 10048–10055, unread, collision-checked. Energy selects. F post hoc. Candidate development only: GR is not patched even if the gate had passed.

**Reject S8.** The preregistered gate is N15 L3 Nov3, n=8: bootstrap 95% CI of mean(E_S8 − E_S4) entirely below 0 **and** median ΔE < 0. Observed mean +0.040, median +0.045, CI [+0.022, +0.057]. The interval lies entirely **above** 0. S8 was lower energy on **0/8** seeds. Fidelity moved with energy (mean ΔF −0.023, 2/8 S8 wins) and is not in the gate.

QAOA agrees: S8 lost energy 0/8, mean ΔE +0.018, CI [+0.009, +0.028] entirely above 0. Under both S4 and S8, QAOA beats Nov3 on energy (mean Δ(E−E0)/N of Nov3−QAOA CI entirely above 0); Nov3 F is mixed. Energy remains the selection rule.

S4 selected start 0 (TQA) on 7/8 seeds for both circuits. S8 still selected start 0 on 7/8 Nov3 and 5/8 QAOA. Extra random starts were almost never the energy winner, and they halved K on the TQA trajectory. Equal-total-budget diversity at this count is a net loss, not a rescue of L=3 fade.

32/32 fits ok. Charged 98304. Outside-cap optimizer queries 0. All 8 B* instances accepted. S4 starts match historical C0. N3COND 772224, LEXACT SPSA 2560, DEPTH C1 327680 unchanged. No GR patch, no continuation retry, no further start-count scan in this report.
