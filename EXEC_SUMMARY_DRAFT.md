# Executive summary (DRAFT, n=88). Rewrite the prose in your own voice before submitting.

Neel rejects LLM-written applications and says he can tell. Treat everything below as your
numbers and structure, not final copy. Rewrite every sentence yourself. The
"where-I-nearly-fooled-myself" paragraph in particular must be genuinely yours.

---

## Suggested title
**J-Lens reads a model's hidden intermediate better than the logit lens, but steers it no
better, and steering-based "validation" is mostly token-injection.**

## Executive summary

**Objective.** J-Lens (the global-workspace paper's Jacobian lens) claims to read a model's
hidden intermediate "working memory." Neel's review argued it is useful for *generating*
hypotheses but unproven for *validating* them, and flagged a specific confound: for "capital
of the champagne country to Paris", maybe editing the hidden `France` step only works because
`France` and `Paris` are linearly related in unembedding space, not because of real multi-hop
mediation. I turn that into a controlled read-vs-write test on **Qwen3-4B**, comparing J-Lens
against a logit-lens and a learned tuned-lens baseline, as both a *reader* (detect the hidden
entity) and a *writer* (steer the answer), with per-item linear-similarity (`linsim`) as the
covariate for Neel's confound. Code, data, and figures are public: [repo link].

**Setup.** 200 multi-hop items (country to capital / currency / language, element to symbol),
filtered to the **88 the model answers zero-shot** (top-8). Three concept directions at
matched norm. J-Lens and the tuned lens are baked/fit on a held-out corpus, never the eval
items. All effects carry bootstrap 95% CIs and paired permutation tests.

**Results.**
- **Reading (positive):** J-Lens ranks the hidden entity best, MRR **0.500 [0.42, 0.58]** vs
  logit **0.420 [0.35, 0.50]**, paired **p=0.051**. It converged there as I added data (p=0.19
  at n=17, 0.07 at n=46, 0.051 at n=88), and it **decisively beats the tuned lens**
  (0.146, p<0.001). So J-Lens is a genuinely better reader of the intermediate.
- **Writing (null vs baseline):** steering with J-Lens moves **0.130** of the probability onto
  the counterfactual answer vs logit lens's **0.124** (paired **p=0.55**, difference CI
  [-0.013, +0.027]). No causal advantage over the cheap baseline; random control ~0.000.
- **The interesting finding:** the steering is mostly token-injection, not multi-hop routing.
  Direct answer-steering dominates entity-steering **~3.6x** (0.46 vs 0.13), and injecting the
  entity direction raises the entity's *own* token about as much as it raises the answer
  (0.146). So using a reading lens as a steering vector to "validate" it mostly measures the
  direction writing its own token back at the read-out.
- **Confound verdict:** the (J-Lens minus logit) advantage is flat across `linsim` (slope
  **-0.05, CI [-0.16, +0.05]**), so the reading advantage is **not** the linear-unembedding
  shortcut Neel raised.

**Scope / limitations.** Single-token J-Lens only; this variant is close to a linearized logit
lens, so the causal tie may be partly expected, and the multi-token / future-token Jacobian is
untested. Same-position steering cannot fully separate injection from mediation (the
token-injection number measures that concern, it does not remove it); cross-position patching
is the next step. n=88 after a top-8 competence filter. Random control passes (~0).

**Where I nearly fooled myself.** [Write this yourself. Raw material: run 1 looked like a
strong positive until I read the raw records and saw the "flip" was a large steering strength
destroying the correct answer, not redirecting it, so I rebuilt the metric to be
suppression-proof; and adding CIs turned a detection "win" into a near-tie that only firmed up
to p=0.05 once I scaled the data. Say it in your own words.]

---

# Where to put the figures (you have 8 current figures)

**In the executive summary / main write-up (the 3 hero figures, in this order):**
1. `figures/figA_reading.png` -> right under "Reading (positive)". The reader result.
2. `figures/figB_steering.png` -> right under "Writing (null vs baseline)". The writer null.
3. `figures/figC_mechanism.png` -> right under "the interesting finding". The token-injection story.

**One supporting figure in the body (the confound, since it answers Neel's specific worry):**
4. `figures/fig3_confound.png` -> right under "Confound verdict". Flat slope vs linsim.

**Appendix / supplementary (reference by name, do not spend body space on them):**
5. `figures/fig1_detection.png` -> per-layer detection MRR with CI bands (shows L* choice and
   that the reader gap is late-layer). Good appendix support for figure A.
6. `figures/fig2_causal_vs_linsim.png` -> per-item causal effect vs linsim for all three
   lenses (the raw scatter behind figure C / the confound).
7. `figures/fig4_controls.png` -> concept vs answer-swap vs random bar (an alternate view of
   figure C; you can cut this if figC is in, to avoid redundancy).
8. `figures/fig5_alpha_curve.png` -> effect vs steering strength; a one-line footnote
   justifying why the headline uses alpha=0.5 (gentlest perturbation). Footnote only.

(Note: the old combined `fig0_headline.png` was deleted; figures A/B/C replace it.)

**Randomly-selected raw examples** (put right after the exec summary, per Neel's guidance):
pull the first 5 kept items by id from `results/steering.json` (jlens, concept, alpha=0.5) and
show for each: prompt, entity->cf_entity, `toward_Ap` (answer moved) vs `push_Ip` (entity
token moved), and one sentence in your words on whether the flip was real or token-injection.
