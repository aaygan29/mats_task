"""Figures. All read the json artifacts and write PNGs to figures/.
Effect metric = toward_Ap (probability mass moved onto the counterfactual answer),
coherent trials only. `alpha` is passed in (chosen a priori = gentlest perturbation)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EFFECT = "toward_Ap"
C = {"jlens": "#1f77b4", "logit": "#d62728", "tuned": "#2ca02c", "random": "#7f7f7f"}
DETECT = ("jlens", "logit", "tuned")


def _coh(records):
    return [r for r in records if r.get("coherent", True)]


def fig1_detection(summary_det, layers, Lstar, path, title_tag=""):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in DETECT:
        ys = [summary_det[m].get(str(l), summary_det[m].get(l)) for l in layers]
        ax.plot(layers, ys, "-o", color=C[m], label=m, lw=2, ms=4)
    ax.axvline(Lstar, ls="--", color="k", alpha=0.5, label=f"L* = {Lstar}")
    ax.set_xlabel("layer"); ax.set_ylabel("MRR of true hidden entity (higher=better)")
    ax.set_title(f"C0 detection: which lens finds the hidden entity?{title_tag}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig2_causal_vs_linsim(records, path, title_tag="", alpha=0.5):
    rc = _coh(records)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for m in ("jlens", "logit", "tuned"):
        pts = [(r["linsim"], r[EFFECT]) for r in rc
               if r["method"] == m and r["mode"] == "concept" and r["alpha"] == alpha]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, color=C[m], label=m, alpha=0.75, s=28)
        if len(set(xs)) >= 2:
            b, c = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, b * xx + c, color=C[m], lw=1.6, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("linsim = cos(U[answer], U[entity])  (low = answer NOT readable from entity)")
    ax.set_ylabel("P(counterfactual answer) moved  [coherent]")
    ax.set_title(f"C2/C3: causal flip vs linearity{title_tag}  (alpha={alpha})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig3_confound(records, path, title_tag="", alpha=0.5):
    by_id = {}
    for r in _coh(records):
        if r["mode"] != "concept" or r["alpha"] != alpha:
            continue
        by_id.setdefault(r["id"], {})[r["method"]] = (r["linsim"], r[EFFECT])
    xs, ys = [], []
    for d in by_id.values():
        if "jlens" in d and "logit" in d:
            xs.append(d["jlens"][0]); ys.append(d["jlens"][1] - d["logit"][1])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.scatter(xs, ys, color="#6a3d9a", s=34)
    if len(set(xs)) >= 2:
        b, c = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, b * xx + c, color="#6a3d9a", lw=1.8, label=f"slope={b:+.2f}, intercept={c:+.2f}")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("linsim = cos(U[answer], U[entity])")
    ax.set_ylabel("J-Lens effect  -  logit-lens effect")
    ax.set_title(f"C3: J-Lens minus logit-lens causal effect vs linearity{title_tag}\n"
                 "points near 0 at all linsim => J-Lens no better than logit lens as a lever")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig4_controls(records, path, title_tag="", alpha=0.5):
    rc = _coh(records)
    cats = ["concept", "answer", "random"]
    vals, errs = [], []
    for mode in cats:
        m = "jlens" if mode != "random" else "random"
        e = [r[EFFECT] for r in rc if r["alpha"] == alpha and r["mode"] == mode and r["method"] == m]
        vals.append(np.mean(e) if e else 0.0)
        errs.append(np.std(e) / max(len(e), 1) ** 0.5 if e else 0.0)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(cats, vals, yerr=errs, color=["#1f77b4", "#ff7f0e", "#7f7f7f"], capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean P(A') moved  [coherent]")
    ax.set_title(f"Controls (J-Lens){title_tag}\nconcept vs answer-swap (fig-15) vs random")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig5_alpha_curve(records, path, title_tag=""):
    """Effect + coherent-N vs alpha per method: shows the effect only lives at small alpha."""
    rc = _coh(records)
    alphas = sorted({r["alpha"] for r in records})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ("jlens", "logit", "tuned"):
        ys, ns = [], []
        for a in alphas:
            e = [r[EFFECT] for r in rc if r["method"] == m and r["mode"] == "concept" and r["alpha"] == a]
            ys.append(np.mean(e) if e else 0.0); ns.append(len(e))
        ax.plot(alphas, ys, "-o", color=C[m], label=f"{m}", lw=2, ms=5)
        for a, y, n in zip(alphas, ys, ns):
            ax.annotate(f"n={n}", (a, y), fontsize=7, alpha=0.6)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("steering strength alpha (x residual norm)")
    ax.set_ylabel("mean P(A') moved  [coherent concept trials]")
    ax.set_title(f"Effect vs steering strength{title_tag}\n(coherent N falls as alpha breaks the model)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)
