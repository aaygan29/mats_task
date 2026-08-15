"""Three separate, readable headline figures (Qwen3-4B).
  figA_reading.png    - detection MRR, three lenses, with CIs + paired-test brackets
  figB_steering.png   - causal effect, three lenses + random, with CIs + paired test
  figC_mechanism.png  - entity-swap vs answer-swap vs token-push vs random
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
        return (float(np.mean(vals)) if len(vals) else 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    m = float(np.mean(vals))
    return m, m - np.percentile(bs, 2.5), np.percentile(bs, 97.5) - m


def sig(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    if p < 0.1:
        return f"p={p:.2f} (trend)"
    return f"p={p:.2f} (n.s.)"


def bracket(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y * 1.03, y * 1.03, y], lw=1.2, c="k")
    ax.text((x1 + x2) / 2, y * 1.045, text, ha="center", va="bottom", fontsize=11)


def load():
    h = json.load(open("results/summary.json"))["headline"]
    recs = json.load(open("results/steering.json"))
    a = h["chosen_alpha"]
    coh = [r for r in recs if r.get("coherent") and r["alpha"] == a]
    return h, coh


def eff(coh, method, mode):
    return [r["toward_Ap"] for r in coh if r["method"] == method and r["mode"] == mode]


def fig_reading(h):
    s = h["C0_detection_stats"]; n = h["n_items_kept"]
    ms = ["jlens", "logit", "tuned"]
    vals = [s["MRR"][m] for m in ms]
    err = [[s["MRR"][m] - s["MRR_CI"][m][0] for m in ms],
           [s["MRR_CI"][m][1] - s["MRR"][m] for m in ms]]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.bar(["J-Lens", "logit lens", "tuned lens"], vals, yerr=err,
           color=[C[m] for m in ms], capsize=6, alpha=0.9, width=0.6)
    top = max(s["MRR_CI"][m][1] for m in ms)
    bracket(ax, 0, 1, top * 1.07, sig(s["jlens_vs_logit_paired_p"]))
    bracket(ax, 0, 2, top * 1.22, sig(s["jlens_vs_tuned_paired_p"]))
    ax.set_ylim(0, top * 1.45)
    ax.set_ylabel("MRR of the hidden entity   (higher = reads it better)", fontsize=11)
    ax.set_title(f"Reading the hidden step (detection)\nQwen3-4B, n={n}, at best layer L*={h['Lstar']}",
                 fontsize=12.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig("figures/figA_reading.png", dpi=150); plt.close(fig)


def fig_steering(h, coh):
    n = h["n_items_kept"]
    order = ["jlens", "logit", "tuned", "random"]
    modes = {"jlens": "concept", "logit": "concept", "tuned": "concept", "random": "random"}
    v, lo, hi = [], [], []
    for m in order:
        mm, l, hh = boot_ci(eff(coh, m, modes[m])); v.append(mm); lo.append(l); hi.append(hh)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.bar(["J-Lens", "logit lens", "tuned lens", "random"], v, yerr=[lo, hi],
           color=[C[m] for m in order], capsize=6, alpha=0.9, width=0.6)
    ax.axhline(0, c="k", lw=0.8)
    top = max(a + b for a, b in zip(v, hi))
    bracket(ax, 0, 1, top * 1.10, sig(h["C2_jlens_minus_logit_paired_p"]))
    ax.set_ylim(min(0, min(v) - 0.02), top * 1.4)
    ax.set_ylabel("P(counterfactual answer) moved   (higher = steers it better)", fontsize=11)
    ax.set_title(f"Steering the answer (causal)\nQwen3-4B, n={n}, alpha={h['chosen_alpha']}",
                 fontsize=12.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig("figures/figB_steering.png", dpi=150); plt.close(fig)


def fig_mechanism(h, coh):
    n = h["n_items_kept"]
    cats = ["entity-swap\n(the real\nmulti-hop test)", "answer-swap\n(steer answer\ndirectly)",
            "entity-token\npush", "random"]
    data = [boot_ci(eff(coh, "jlens", "concept")), boot_ci(eff(coh, "jlens", "answer")),
            (h["token_push_jlens_concept"], 0, 0), boot_ci(eff(coh, "random", "random"))]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.bar(cats, [d[0] for d in data], yerr=[[d[1] for d in data], [d[2] for d in data]],
           color=["#1f77b4", "#ff7f0e", "#9467bd", "#7f7f7f"], capsize=6, alpha=0.9, width=0.62)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("effect size (probability mass)", fontsize=11)
    ax.set_title(f"Is the J-Lens steering REAL multi-hop routing?\n"
                 f"answer-swap dominates the real test; token-push is a big share (n={n})",
                 fontsize=12)
    ax.grid(alpha=0.25, axis="y"); ax.tick_params(axis="x", labelsize=9.5)
    fig.tight_layout(); fig.savefig("figures/figC_mechanism.png", dpi=150); plt.close(fig)


def main():
    h, coh = load()
    fig_reading(h); fig_steering(h, coh); fig_mechanism(h, coh)
    print("wrote figures/figA_reading.png, figB_steering.png, figC_mechanism.png")


if __name__ == "__main__":
    main()
