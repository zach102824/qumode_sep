import json, os
import numpy as np

ROOT = "/workspace/qumode_sep/oct_explore"
d4 = json.load(open(os.path.join(ROOT, "results/clean16/diagnosis_data.json")))
d8 = json.load(open(os.path.join(ROOT, "results/clean16_8sat/diagnosis_data.json")))

def b90(arr):
    arr = np.sort(np.asarray(arr, dtype=np.float64))
    return float(np.mean(arr[int(0.9 * len(arr)):]))

def med(arr):
    return float(np.median(np.asarray(arr, dtype=np.float64)))

def mean(arr):
    return float(np.mean(np.asarray(arr, dtype=np.float64)))

def summarize_sat(d, label):
    print(f"================ {label} ================")
    for n_str in ["8", "12", "16", "20"]:
        n = int(n_str)
        rows = d[n_str]
        print(f"\n--- {label} n = {n} (2^n = {1<<n}) ---")
        # 1. Smearing vs locking across Q, G, QC_r0, N_r0, N_pv_r0
        print("Method | Med 1/p | B90 1/p | Mean p_sol | Med H(P)/n | Med N_eff | Med P_top1_w | Med P_top10_w | Med P_top100_w | Ratio P_top10_w/p_sol (Med / Hard10)")
        for cname in ["Q", "G", "QC_block", "QC_r0", "N_r0", "N_pv_r0", "N_corr_r0", "N_grover_r0", "N_unif_dSdB", "N_unif_w"]:
            invp = [r["circuits"][cname]["inv_p"] for r in rows]
            psol = [r["circuits"][cname]["p_sol"] for r in rows]
            ent_n = [r["circuits"][cname]["ent"] / n for r in rows]
            neff = [r["circuits"][cname]["neff"] for r in rows]
            t1w = [r["circuits"][cname]["p_top1_wrong"] for r in rows]
            t10w = [r["circuits"][cname]["p_top10_wrong"] for r in rows]
            t100w = [r["circuits"][cname]["p_top100_wrong"] for r in rows]
            rat = [r["circuits"][cname]["p_top10_wrong"] / max(r["circuits"][cname]["p_sol"], 1e-12) for r in rows]
            idx_sort = np.argsort(invp)
            hard_idx = idx_sort[90:]
            rat_hard = np.median([rat[i] for i in hard_idx])
            t10w_hard = np.median([t10w[i] for i in hard_idx])
            neff_hard = np.median([neff[i] for i in hard_idx])
            print(f"{cname:11s} | {med(invp):8.2f} | {b90(invp):9.2f} | {mean(psol):9.5f} | {med(ent_n):10.4f} | {med(neff):9.1f} (H:{neff_hard:7.1f}) | {med(t1w):12.5f} | {med(t10w):13.5f} (H:{t10w_hard:7.4f}) | {med(t100w):14.5f} | {med(rat):8.2f}x / {rat_hard:8.2f}x")

        # 2. Ghost minima & projection stats
        vk_blk = mean([r["proj_block"]["var_kept"] for r in rows])
        vk_r0 = mean([r["proj_r"][0]["var_kept"] for r in rows])
        cr_r0 = mean([r["proj_r"][0]["corr_PE_E"] for r in rows])
        amin_sol = mean([r["proj_r"][0]["argmin_is_sol"] for r in rows])
        e_amin = mean([r["proj_r"][0]["E_at_argmin"] for r in rows])
        ng_u = med([r["proj_r"][0]["n_ghost_unif"] for r in rows])
        ng_u_m = mean([r["proj_r"][0]["n_ghost_unif"] for r in rows])
        ng_pv = med([r["proj_r"][0]["n_ghost_pv"] for r in rows])
        ng_eff = med([r["proj_r"][0]["n_ghost_eff"] for r in rows])
        print(f"Projection r=0: VarKept(blk/r0)={vk_blk:.4f}/{vk_r0:.4f}, Corr(PE,E)={cr_r0:.4f}, argmin_is_sol={amin_sol*100:.1f}%, mean E(argmin)={e_amin:.2f}, Med N_ghost(unif/pv/eff)={ng_u:.1f}/{ng_pv:.1f}/{ng_eff:.1f} (mean unif={ng_u_m:.1f})")

        # Top wrong state properties for Q, QC_r0, N_r0, N_pv_r0
        for cname in ["Q", "QC_r0", "N_r0", "N_pv_r0"]:
            t1_E = mean([r["circuits"][cname]["top1_E"] for r in rows])
            t10_E = mean([r["circuits"][cname]["top10_wmean_E"] for r in rows])
            t1_pez = mean([r["circuits"][cname]["top1_PE_z"] for r in rows])
            beats_u = mean([r["circuits"][cname]["top1_beats_sol_unif"] for r in rows]) * 100
            beats_pv = mean([r["circuits"][cname]["top1_beats_sol_pv"] for r in rows]) * 100
            h_sol = mean([r["circuits"][cname]["top1_ham_sol"] for r in rows])
            cw = mean([r["circuits"][cname]["top1_cav_wrong"] for r in rows])
            bpw = mean([r["circuits"][cname]["top1_bpw_cav"] for r in rows])
            h_pv = mean([r["circuits"][cname]["top1_ham_pv"] for r in rows])
            h_E1 = mean([r["circuits"][cname]["top1_ham_E1"] for r in rows])
            h_am = mean([r["circuits"][cname]["top1_ham_argmin_PE"] for r in rows])
            e_sh = np.mean([r["circuits"][cname]["e_shells"] for r in rows], axis=0)
            print(f"  TopWrong {cname:7s}: E_top1={t1_E:.2f}, E_top10w={t10_E:.2f}, PE_z={t1_pez:.2f}, beats_sol(u/pv)={beats_u:.0f}%/{beats_pv:.0f}%, dH(sol)={h_sol:.2f} (cav_w={cw:.2f}, bit/cav={bpw:.2f}), dH(pv/E1/amin)={h_pv:.2f}/{h_E1:.2f}/{h_am:.2f}, E_shells(0,1,2,3,4,5+)=[{','.join(f'{x:.3f}' for x in e_sh)}]")

        # Pv & E1 stats
        pv_acc = mean([r["pv_acc"] for r in rows]) * 100
        pv_conf = mean([r["pv_conf"] for r in rows])
        h_pv_s = mean([r["ham_pv_sol"] for r in rows])
        e_pv = mean([r["E_at_pv"] for r in rows])
        h_e1_s = mean([r["ham_E1_sol"][0] for r in rows])
        e_e1 = mean([r["E_at_E1"][0] for r in rows])
        smk = mean([r["sol_in_mark"][0] for r in rows]) * 100
        print(f"  GNN cues: pv_acc={pv_acc:.1f}%, pv_conf={pv_conf:.3f}, dH(pv,sol)={h_pv_s:.2f}, E(pv)={e_pv:.2f} | dH(E1_r0,sol)={h_e1_s:.2f}, E(E1_r0)={e_e1:.2f}, sol_cavs_in_grover_mark={smk:.1f}%")

        # Layer-by-layer for Q, QC_r0, N_r0, N_pv_r0
        for cname in ["Q", "QC_r0", "N_r0", "N_pv_r0"]:
            ps_l = np.median([r["circuits"][cname]["p_sol_l"] for r in rows], axis=0)
            nf_l = np.median([r["circuits"][cname]["neff_l"] for r in rows], axis=0)
            mE_l = np.mean([r["circuits"][cname]["mE_l"] for r in rows], axis=0)
            mPE_l = np.mean([r["circuits"][cname]["mPE_l"] for r in rows], axis=0)
            t10_l = np.median([r["circuits"][cname]["top10_l"] for r in rows], axis=0)
            print(f"  Layers {cname:7s}: med_p_sol={[round(x,5) for x in ps_l]} | med_Neff={[round(x,1) for x in nf_l]} | mean_E={[round(x,2) for x in mE_l]} | mean_PEz={[round(x,2) for x in mPE_l]} | med_Ptop10w={[round(x,4) for x in t10_l]}")

        # Across R=4 placements
        for fam in ["QC", "N_full", "N_pv"]:
            inv_rr = med([r["placement_comp"][fam]["inv_p_rr"] for r in rows])
            b90_rr = b90([r["placement_comp"][fam]["inv_p_rr"] for r in rows])
            inv_r0 = med([1.0/max(r["placement_comp"][fam]["p_r"][0], 1e-12) for r in rows])
            inv_b = med([1.0/max(r["placement_comp"][fam]["p_max"], 1e-12) for r in rows])
            sp = med([r["placement_comp"][fam]["spread"] for r in rows])
            nd = mean([r["placement_comp"][fam]["n_distinct_top1"] for r in rows])
            pwh = mean([r["placement_comp"][fam]["mean_pw_ham_top1"] for r in rows])
            ovl = mean([r["placement_comp"][fam]["mean_pw_top10_overlap"] for r in rows]) * 100
            usz = mean([r["placement_comp"][fam]["top10_union_size"] for r in rows])
            print(f"  R=4 {fam:6s}: Med 1/p (r0={inv_r0:.2f}, RR={inv_rr:.2f}, best_r={inv_b:.2f}, B90_RR={b90_rr:.2f}), Med spread(max/min)={sp:.2f}x, distinct_top1={nd:.2f}/4, pw_dH(top1)={pwh:.2f}, top10_ovl={ovl:.1f}%, union={usz:.1f}/40")

summarize_sat(d4, "4-SAT")
summarize_sat(d8, "8-SAT")
