"""Single headline figure summarizing the whole project (n=46, Qwen3-4B).
3 panels: (A) reading the hidden step, (B) steering the answer, (C) is the steering real?
Reads results/summary.json + results/steering.json. Run: python -m src.plot_headline"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"jlens": "#1f77b4", "logit": "#d62728", "tuned": "#2ca02c", "random": "#7f7f7f"}


def boot_ci(vals, n=3000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return (np.mean(vals) if len(vals) else 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    m = float(np.mean(vals))
    return m, m - np.percentile(bs, 2.5), np.percentile(bs, 97.5) - m


def sig(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return f"p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    if p < 0.1:
        return f"p={p:.2f} (trend)"
    return f"p={p:.2f} (n.s.)"


def bracket(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y * 1.03, y * 1.03, y], lw=1.1, c="k")
    ax.text((x1 + x2) / 2, y * 1.05, text, ha="center", va="bottom", fontsize=8.5)


def main():
    h = json.load(open("results/summary.json"))["headline"]
    recs = json.load(open("results/steering.json"))
    a = h["chosen_alpha"]
    coh = [r for r in recs if r.get("coherent") and r["alpha"] == a]

    def eff(method, mode):
        return [r["toward_Ap"] for r in coh if r["method"] == method and r["mode"] == mode]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5.2))
    n = h["n_items_kept"]
    fig.suptitle(f"Is J-Lens a better interpretability signal than the cheap baseline?  "
                 f"Qwen3-4B, n={n} multi-hop items, L*={h['Lstar']}", fontsize=13, y=0.99)

    # ---- Panel A: reading (detection MRR) ----
    s = h["C0_detection_stats"]
    ms = ["jlens", "logit", "tuned"]
    vals = [s["MRR"][m] for m in ms]
    err = [[s["MRR"][m] - s["MRR_CI"][m][0] for m in ms],
           [s["MRR_CI"][m][1] - s["MRR"][m] for m in ms]]
    axA.bar(ms, vals, yerr=err, color=[C[m] for m in ms], capsize=5, alpha=0.9)
    top = max(s["MRR_CI"][m][1] for m in ms)
    bracket(axA, 0, 1, top * 1.06, sig(s["jlens_vs_logit_paired_p"]))
    bracket(axA, 0, 2, top * 1.20, sig(s["jlens_vs_tuned_paired_p"]))
    axA.set_ylim(0, top * 1.42)
    axA.set_ylabel("MRR of the hidden entity  (higher = reads it better)")
    axA.set_title("A. READING the hidden step\nJ-Lens trends above logit lens, clearly beats tuned")
    axA.grid(alpha=0.25, axis="y")

    # ---- Panel B: writing (causal effect) ----
    order = ["jlens", "logit", "tuned", "random"]
    modes = {"jlens": "concept", "logit": "concept", "tuned": "concept", "random": "random"}
    bvals, blo, bhi = [], [], []
    for m in order:
        mm, lo, hi = boot_ci(eff(m, modes[m]))
        bvals.append(mm); blo.append(lo); bhi.append(hi)
    axB.bar(order, bvals, yerr=[blo, bhi], color=[C[m] for m in order], capsize=5, alpha=0.9)
    axB.axhline(0, c="k", lw=0.8)
    topB = max(v + e for v, e in zip(bvals, bhi))
    bracket(axB, 0, 1, topB * 1.10, sig(h["C2_jlens_minus_logit_paired_p"]))
    axB.set_ylim(min(0, min(bvals) - 0.02), topB * 1.35)
    axB.set_ylabel("P(counterfactual answer) moved  (higher = steers it better)")
    axB.set_title("B. STEERING the answer\nJ-Lens = logit lens (well-powered null); random ~ 0")
    axB.grid(alpha=0.25, axis="y")

    # ---- Panel C: is the steering real? ----
    cats = ["entity-swap\n(the real test)", "answer-swap\n(direct, cheat)",
            "entity-token\npush", "random"]
    cv = [boot_ci(eff("jlens", "concept")), boot_ci(eff("jlens", "answer")),
          (h["token_push_jlens_concept"], 0, 0), boot_ci(eff("random", "random"))]
    axC.bar(cats, [c[0] for c in cv], yerr=[[c[1] for c in cv], [c[2] for c in cv]],
            color=["#1f77b4", "#ff7f0e", "#9467bd", "#7f7f7f"], capsize=5, alpha=0.9)
    axC.axhline(0, c="k", lw=0.8)
    axC.set_ylabel("effect size (prob mass)")
    axC.set_title("C. Is the steering REAL multi-hop?\nanswer-swap dominates ~4x; token-push ~ the effect")
    axC.grid(alpha=0.25, axis="y")
    axC.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/fig0_headline.png", dpi=150)
    print("wrote figures/fig0_headline.png")


if __name__ == "__main__":
    main()
