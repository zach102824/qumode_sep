"""Runtime mutations of noisy_simulation (source text re-exec'd in memory; repo files untouched)."""
import inspect, textwrap, sys
sys.path.insert(0, '/workspace/qumode_sep')
from noisy_simulation import circuit, kraus, device
def patch(mod, name, pairs, also=()):
    src = textwrap.dedent(inspect.getsource(getattr(mod, name)))
    for old, new in pairs:
        assert old in src, (name, old); src = src.replace(old, new)
    exec(compile(src, f"<mutated {name}>", "exec"), mod.__dict__)
    for m in also: setattr(m, name, mod.__dict__[name])
def M_gamma2(): device.Device.gamma = property(lambda s: 2.0 * s.noise_scale / s.t1_q)
def M_gphi_no_half(): patch(kraus, "qubit_kraus", [("lam = np.exp(-float(gpt))", "lam = np.exp(-2.0 * float(gpt))")], [circuit])
def M_wrong_cavity():
    patch(circuit, "cavity_idle", [("for ax in axes:", "for ax in (axes[0], axes[0]):")])
    patch(circuit, "jp_cavity_channel", [("_x_cavity_loss(X, w, 3, 5, 7)", "_x_cavity_loss(X, w, 2, 4, 6)")])
def M_drop_wait2():
    patch(circuit, "jp_direct", [("r = apply_unitary(r, JP_MID, 0)\n    r = _wait_direct(r, dev)", "r = apply_unitary(r, JP_MID, 0)\n    r = _wait_direct(r, dev.with_(noise_scale=0.0))")])
    patch(circuit, "jp_cavity_channel", [("X = q_op(X, superop([JP_MID]))\n    X = wait(X)", "X = q_op(X, superop([JP_MID]))\n    for _s in range(dev.n_slice):\n        X = phase(X)")])
def M_idle_twice():
    patch(circuit, "noisy_layer", [("get_cavity_channel(dev, nf, jumps, dev.tau_ecd)", "get_cavity_channel(dev, nf, jumps, 2 * dev.tau_ecd)"),
        ("transmon_idle(rho, dev, dev.tau_ecd + dev.tau_jp)", "transmon_idle(rho, dev, 2 * dev.tau_ecd + dev.tau_jp)"),
        ("rho = cavity_idle(rho, dev, dev.tau_ecd)", "rho = cavity_idle(cavity_idle(rho, dev, dev.tau_ecd), dev, dev.tau_ecd)"),
        ("rho = transmon_idle(rho, dev, dev.tau_ecd)\n", "rho = transmon_idle(transmon_idle(rho, dev, dev.tau_ecd), dev, dev.tau_ecd)\n")])
def M_qubit_adjoint(): patch(kraus, "qubit_kraus", [("return [K for K in out if", "return [K.conj().T for K in out if")], [circuit])
def M_cavity_adjoint():   # photon gain instead of loss (shift form + dense form); fast-path propagation untouched
    patch(kraus, "apply_cavity_loss", [("out[_ax(N, k, slice(0, m))] += wk * wb * rho[_ax(N, k, slice(j, nf))]", "out[_ax(N, k, slice(j, nf))] += wk * wb * rho[_ax(N, k, slice(0, m))]")], [circuit])
    patch(kraus, "shift_to_dense", [("K[col - j, col] = w[col]", "K[col, col - j] = w[col]")])
def M_sign_exponent():
    patch(kraus, "cavity_loss_weights", [("- 0.5 * kt * (m - j))", "+ 0.5 * kt * (m - j))"), ("w0 = np.exp(-0.5 * kt * n)", "w0 = np.exp(+0.5 * kt * n)")], [circuit])
def M_jmax1():
    patch(kraus, "cavity_loss_weights", [("    kt = float(kt)\n", "    jmax = 1 if (jmax is None or jmax > 1) else jmax\n    kt = float(kt)\n")], [circuit])
def M_transmon_wrong_mode(): patch(circuit, "transmon_idle", [("for ax in axes:", "for ax in (axes[0],) * len(axes):")])
def M_kappa_over_2pi(): device.Device.kappa = property(lambda s: s.noise_scale / s.t1_cav / (2 * 3.141592653589793))
MUTS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("M_")}
def apply(name):
    if name != "none": MUTS[name]()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.savefig = lambda *a, **k: None
