"""Figures. All read the json artifacts and write PNGs to figures/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"jlens": "#1f77b4", "logit": "#d62728", "probe": "#2ca02c", "random": "#7f7f7f"}


def fig1_detection(summary_det, layers, Lstar, path, title_tag=""):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in ("jlens", "logit"):
        ys = [summary_det[m][str(l)] if str(l) in summary_det[m] else summary_det[m][l]
              for l in layers]
        ax.plot(layers, ys, "-o", color=C[m], label=m, lw=2, ms=4)
    ax.axvline(Lstar, ls="--", color="k", alpha=0.5, label=f"L* = {Lstar}")
    ax.set_xlabel("layer"); ax.set_ylabel("MRR of true hidden entity (higher=better)")
    ax.set_title(f"C0 detection: does the lens find the hidden entity?{title_tag}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


EFFECT = "toward_Ap"   # probability mass moved onto the counterfactual answer


def _coh(records):
    return [r for r in records if r.get("coherent", True)]


def _concept_best_alpha(records):
    """Largest mean toward_Ap for J-Lens concept-swap among COHERENT trials."""
    rc = _coh(records)
    alphas = sorted({r["alpha"] for r in rc}) or sorted({r["alpha"] for r in records})
    best, best_score = alphas[0], -1e9
    for a in alphas:
        vals = [r[EFFECT] for r in rc
                if r["alpha"] == a and r["mode"] == "concept" and r["method"] == "jlens"]
        sc = np.mean(vals) if vals else -1e9
        if sc > best_score:
            best_score, best = sc, a
    return best


def fig2_causal_vs_linsim(records, path, title_tag=""):
    a = _concept_best_alpha(records)
    rc = _coh(records)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ("jlens", "logit"):
        pts = [(r["linsim"], r[EFFECT]) for r in rc
               if r["method"] == m and r["mode"] == "concept" and r["alpha"] == a]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, color=C[m], label=m, alpha=0.75, s=30)
        if len(xs) >= 2:
            b, c = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, b * xx + c, color=C[m], lw=1.6, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("linsim = cos(U[answer], U[entity])  (low = answer NOT readable from entity)")
    ax.set_ylabel("P(counterfactual answer) moved  [coherent trials]")
    ax.set_title(f"C2/C3: does J-Lens move mass onto A', esp. at low linsim?{title_tag}\n(alpha={a})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig3_confound(records, path, title_tag=""):
    a = _concept_best_alpha(records)
    by_id = {}
    for r in _coh(records):
        if r["mode"] != "concept" or r["alpha"] != a:
            continue
        by_id.setdefault(r["id"], {})[r["method"]] = (r["linsim"], r[EFFECT])
    xs, ys = [], []
    for d in by_id.values():
        if "jlens" in d and "logit" in d:
            xs.append(d["jlens"][0]); ys.append(d["jlens"][1] - d["logit"][1])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(xs, ys, color="#6a3d9a", s=34)
    if len(xs) >= 2:
        b, c = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, b * xx + c, color="#6a3d9a", lw=1.8,
                label=f"slope={b:+.2f} (want >=0)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("linsim = cos(U[answer], U[entity])")
    ax.set_ylabel("J-Lens effect  -  logit-lens effect")
    ax.set_title(f"C3 the decisive test: J-Lens advantage vs linearity{title_tag}\n"
                 "advantage persisting at low linsim => real mediation, not artifact")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)


def fig4_controls(records, path, title_tag=""):
    """Concept-swap vs answer-swap (Neel's dominance control) vs random, for J-Lens."""
    a = _concept_best_alpha(records)
    cats = ["concept", "answer", "random"]
    rc = _coh(records)
    vals, errs = [], []
    for mode in cats:
        m = "jlens" if mode != "random" else "random"
        e = [r[EFFECT] for r in rc
             if r["alpha"] == a and r["mode"] == mode and r["method"] == m]
        vals.append(np.mean(e) if e else 0.0)
        errs.append(np.std(e) / max(len(e), 1) ** 0.5 if e else 0.0)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar(cats, vals, yerr=errs, color=["#1f77b4", "#ff7f0e", "#7f7f7f"], capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean P(A') moved  [coherent]")
    ax.set_title(f"Controls (J-Lens){title_tag}\nconcept-swap should beat answer-swap & random")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(path, dpi=140)
    plt.close(fig)
