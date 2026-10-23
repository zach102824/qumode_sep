with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_prep = 'def prepare_split(split_name, max_count=None):'
new_prep = 'def prepare_split(split_name, max_count=None, r_branch=None):'
assert old_prep in code
code = code.replace(old_prep, new_prep)

old_h = '    h_bp = BP_DATA[f"{split_name}_h_dec"] if a.relabel in ("bpdec", "bpcenter") else BP_DATA[f"{split_name}_h_bp"]'
new_h = '''    if r_branch is not None:
        h_bp = BP_DATA[f"{split_name}_h_bp4"][r_branch]
    else:
        h_bp = BP_DATA[f"{split_name}_h_dec"] if a.relabel in ("bpdec", "bpcenter") else BP_DATA[f"{split_name}_h_bp"]'''
assert old_h in code
code = code.replace(old_h, new_h)

old_end = """    th_best = jnp.asarray(best_val["th"], jnp.float32)
    m_val = eval_metrics(*ev_main(th_best, D_va))
    p_te, lk_te = ev_main(th_best, D_te)
    m_test = eval_metrics(p_te, lk_te)
    p_chk, lk_chk = ev_chk(th_best, D_te)
    m_chk = eval_metrics(p_chk, lk_chk)

    res = {
        "args": vars(a), "ws_ref": WS_REF, "NF": NF, "NF_chk": NF_CHK, "NU": NU,
        "best_step": best_val["step"], "val": m_val, "test": m_test, f"test_NF{NF_CHK}": m_chk,
        "p_test": np.asarray(p_te, float).tolist(), "sec": time.time() - t0
    }"""

new_end = """    th_best = jnp.asarray(best_val["th"], jnp.float32)
    np.savez_compressed(OUT.replace(".json", "_theta.npz"), theta=np.asarray(th_best, np.float32))
    m_val = eval_metrics(*ev_main(th_best, D_va))
    p_te, lk_te = ev_main(th_best, D_te)
    m_test = eval_metrics(p_te, lk_te)
    p_chk, lk_chk = ev_chk(th_best, D_te)
    m_chk = eval_metrics(p_chk, lk_chk)

    if a.relabel in ("bpdec", "bpcenter", "bppol"):
        ps_r, lks_r = [np.asarray(p_chk, float)], [np.asarray(lk_chk, float)]
        for r_b in [1, 2, 3]:
            D_r = prepare_split("test", r_branch=r_b)
            pr_i, lk_i = ev_chk(th_best, D_r)
            ps_r.append(np.asarray(pr_i, float)); lks_r.append(np.asarray(lk_i, float))
        ps_r = np.stack(ps_r, axis=0)
        lks_r = np.stack(lks_r, axis=0)
        # Uniform RR4 and weighted RR4 (0.4, 0.2, 0.2, 0.2) and Best-of-4
        m_rr4 = eval_metrics(np.mean(ps_r, axis=0), np.mean(lks_r, axis=0))
        m_wrr4 = eval_metrics(0.40 * ps_r[0] + 0.20 * ps_r[1] + 0.20 * ps_r[2] + 0.20 * ps_r[3], np.mean(lks_r, axis=0))
        m_best4 = eval_metrics(np.max(ps_r, axis=0), np.mean(lks_r, axis=0))
        for d_tgt in (m_test, m_chk):
            d_tgt["rr4_med"] = m_rr4["median_inv_p"]
            d_tgt["rr4_B90"] = m_rr4["B90_inv_p"]
            d_tgt["wrr4_med"] = m_wrr4["median_inv_p"]
            d_tgt["wrr4_B90"] = m_wrr4["B90_inv_p"]
            d_tgt["best4_med"] = m_best4["median_inv_p"]
            d_tgt["best4_B90"] = m_best4["B90_inv_p"]

    res = {
        "args": vars(a), "ws_ref": WS_REF, "NF": NF, "NF_chk": NF_CHK, "NU": NU,
        "best_step": best_val["step"], "val": m_val, "test": m_test, f"test_NF{NF_CHK}": m_chk,
        "p_test": np.asarray(p_te, float).tolist(), "sec": time.time() - t0
    }"""

assert old_end in code
code = code.replace(old_end, new_end)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Patched RR4/wRR4/Best4 evaluation and theta saving into gnn_5block.py!")
