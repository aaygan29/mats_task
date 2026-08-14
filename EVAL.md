# How to evaluate this project (process + result), and how it maps to Neel's interests

This file is the rubric. It says what "success" means, what each result would imply, and
why the project sits inside Neel Nanda's stated research interests.

## What we are actually claiming (and the decision rule)

The whole project outputs one decisive number: **the slope of `(J-Lens effect - logit-lens
effect)` against `linsim`** (`results/summary.json -> headline.C3_advantage_slope_vs_linsim`),
plus the low-`linsim` effect means.

| Outcome | Interpretation |
|---|---|
| J-Lens effect > logit-lens at low `linsim` (slope >= ~0) | J-Lens is a **genuine causal handle** on the intermediate; it adds value the unembedding cannot. Positive result. |
| J-Lens ~ logit-lens once `linsim -> 0` (slope << 0, advantage only at high `linsim`) | J-Lens' apparent causal power in multi-hop recall is largely the **linear-unembedding shortcut** Neel suspected. Clean negative. |
| Neither flips the answer above the random control | The setting is too weak (model can't mediate, or steering breaks it). Report and diagnose, do not overclaim. |

All three are publishable-for-an-application, because the design makes the answer legible.
A well-analysed negative beats a hyped positive here.

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
