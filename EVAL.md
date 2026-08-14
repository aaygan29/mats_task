# How to evaluate this project (process + result), and how it maps to Neel's interests

This file is the rubric. It says what "success" means, what each result would imply, and
why the project sits inside Neel Nanda's stated research interests.

## What we are actually claiming (and the decision rule)

NOTE (post-hoc, after the run): the project was *designed* around one decisive number — the
slope of `(J-Lens - logit-lens)` effect vs `linsim` (C3). In practice C3 came out
**underpowered and inconclusive** (slope -0.24, 95% CI [-0.74, +0.15] includes 0 at n=17), so
it does NOT adjudicate anything. The claims that actually survived with uncertainty are two
comparisons-to-baseline plus one methodological finding:

| Claim | Verdict | Evidence |
|---|---|---|
| J-Lens is a better **detector** than logit lens (C0) | **not significant** at n=17 | MRR 0.66 vs 0.55, paired p=0.19, CIs overlap; but J-Lens > tuned lens p=0.0002 |
| J-Lens is a better **causal lever** than logit lens (C2) | **null** (robust across layers) | 0.149 vs 0.127, paired p=0.50, diff-CI straddles 0 |
| Steering-based "validation" of a lens is contaminated by **token-injection** | **supported, layer-dependent** | answer-swap dominates entity-swap 3.7x-23x; token_push comparable to answer effect |
| The linsim confound (C3) | **unrefuted, not resolved** | slope CI includes 0 |

A well-analysed null beats a hyped positive here. The token-injection finding is the novel,
transferable part (it applies to logit lens and probes too, not just J-Lens).

## Gate ladder (each rung must pass before the next is meaningful)

- **G0 Provenance** — items are answered zero-shot by the *real* model (`verify.json`,
  `keep_rank<=5`). If the kept set is tiny, say so; a claim about mediation is meaningless on
  prompts the model can't do.
- **G1 Detection (C0)** — J-Lens MRR of the true entity beats logit-lens at some layer; that
  layer is `L*`. If J-Lens can't even detect the intermediate, stop.
- **G2 Mediation (C1)** — concept-swap effect >> random-direction control. Rules out "any big
  edit moves the answer".
- **G3 Real multi-hop, not token-push** — for real flips, `delta_Ap` (answer moved) must not be
  explained by `delta_Ip` (raw swapped-entity token moved). This is the free control the
  two-hop design buys.
- **G4 Baseline (C2)** — J-Lens vs logit-lens vs diff-mean probe, matched norm.
- **G5 Confound (C3)** — the decisive slope test above.
- **G6 Robustness** — re-run with a second seed; random control stays ~0; effect signs stable.

## Sanity checks the applicant must personally do (Neel weights this heavily)

- Read >= 10 raw `steering.json` records and confirm a "flip" is a real move toward the
  counterfactual answer, by eye.
- Read `verify.json` and confirm kept items really are answered correctly.
- Recompute one headline number with a fresh one-liner from the raw records.
- Look at the `dominance_control_answer_swap_jlens`: if answer-swap strictly dominates
  concept-swap everywhere, that is Neel's fig-15 warning firing, and the mediation story is
  weak. Report it honestly.

## Why this is inside Neel's stated interests

- **Applied / pragmatic interpretability, measured against baselines** — the entire design is
  "new method vs cheap baselines on a task that matters (model forensics / auditing)". Logit
  lens and a probe are first-class comparisons, not afterthoughts.
- **Red-teaming a new, promising method** — J-Lens is exactly the kind of newer method he says
  he wants stress-tested for reliability and false positives.
- **Answers his own open question** — his review explicitly asks whether J-Lens is causal or
  just hypothesis-generation, and names the multi-hop linear-relatedness confound. We test it.
- **Simplicity-first** — steering vectors, logit lens, a diff-of-means probe, and one measured
  covariate. No SAE hill-climbing, no toy models.
- **Truth-seeking** — a negative (J-Lens ~ logit lens at low `linsim`) is a real, reportable
  finding, and the harness is built so the applicant can catch the agent being wrong.

## What is explicitly out of scope (say so in the write-up)

Multi-token / future-token J-Lens; full-vocabulary detection; cross-position activation
patching; large-N statistics. These are the honest limitations and the natural next steps.
