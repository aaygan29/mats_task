# Write-up material (facts + structure). REWRITE THE PROSE IN YOUR OWN VOICE.

Neel rejects LLM-written applications. This file gives you the numbers, the structure, and
draft sentences as raw material. Do not paste it. Rewrite every sentence in first person, in
your voice. The judgement-under-uncertainty paragraph in particular must be yours.

---

## Suggested title (one line, top of the Google Doc)
"Does J-Lens read a model's hidden intermediate better than the logit lens, and can you steer
with it? On Qwen3-4B: no significant reading edge, no causal edge, and steering-'validation'
is mostly token-injection."

## Application-FORM summary answers (he reads these FIRST — draft, then rewrite)
- **What did you do?** I tested whether J-Lens (the global-workspace Jacobian lens) is a
  genuinely better interpretability signal than the cheap logit lens, on multi-hop factual
  recall in Qwen3-4B, using J-Lens as both a *reader* (detect a hidden intermediate entity)
  and a *steering vector* (flip the answer). I compared against logit lens and a learned
  tuned lens, with per-item linear-similarity as the covariate for Neel's France/Paris confound.
- **Key experiment.** Matched-norm causal steering of the hidden intermediate at the best
  layer, J-Lens vs logit vs tuned, measuring probability mass moved onto the counterfactual
  answer, with random / answer-swap / token-push controls.
- **Most surprising number.** Using a reading lens as a steering vector to "validate" a
  detected concept is mostly *token-injection*: steering the entity direction re-injects the
  entity's own token, and directly steering the answer beats steering the entity by 3.7x-23x.
- **Biggest limitation.** n=17 after the zero-shot filter, so the detection edge (0.66 vs
  0.55) is not significant (paired p=0.19) and the confound slope CI includes 0. Single-token
  J-Lens; same-position steering.

## Exec summary (<=1 page / 600 words) — bullet structure
- Problem (2 sentences): J-Lens claims to read intermediate "working memory"; Neel's review
  says useful for *generating* hypotheses, unproven for *validating*, and names the multi-hop
  linear-relatedness confound. I turn that into a read-vs-write test on Qwen3-4B.
- What I did (1 sentence): three concept-direction lenses (J-Lens / logit / tuned), detection
  + matched-norm steering, held-out fitting, bootstrap CIs + paired tests.
- Result 1 — detection: J-Lens MRR 0.66 [0.52-0.80] vs logit 0.55 [0.40-0.70], paired p=0.19
  -> numerically best but NOT significant. Beats tuned lens (0.29, p=0.0002), a weak baseline.
- Result 2 — causation: J-Lens 0.149 vs logit 0.127, paired p=0.50, CI straddles 0 -> no edge.
- Result 3 — the finding: steering-'validation' is contaminated by token-injection
  (answer-swap dominates entity-swap 3.7x-23x; token_push comparable to the answer effect).
- Confound verdict: slope -0.24, CI [-0.74,+0.15] includes 0 -> unrefuted, not resolved.
- Why interesting (1 honest sentence, your voice).

## Randomly-selected raw examples (put right after exec summary)
State a fixed rule so it's visibly not cherry-picked, e.g. "the first 5 kept items by id".
Pull from `results/steering.json` (jlens, concept, alpha=0.5). For each: prompt, entity->
cf_entity, answer/cf_answer, `toward_Ap` (answer moved) vs `push_Ip` (entity token moved),
and ONE sentence in your words on whether the flip looked real or was token-injection.
Cross-check kept items in `results/verify.json`.

## The paragraph only you can write (do NOT let an LLM write this)
Describe the moment you distrusted your own result: the first run's metric went positive
because a huge alpha was *destroying* the answer, not steering it; you caught it by reading
raw records; you switched to a probability metric that can't be gamed by suppression; and
later, when you added a proper CI, the detection "win" you'd been about to report turned out
to be inside the noise. That self-correction is the most convincing thing in the application.

## Figures
Hero = fig2 (causal vs linsim). Body = fig1 (detection with CI bands), fig4 (controls =
the token-injection story). Appendix = fig3 (inconclusive slope). Footnote only = fig5.

## Time log (fill your real Toggl; honest non-round numbers read as real)
Reading J-Lens paper + Neel's review ~3h; prompt design + zero-shot filtering ~2.5h;
implementing 3 lenses + steering harness ~4h; run 1 + catching the metric/alpha bug + redesign
~3.5h; main runs + debug ~2h; reading raw records + catching token-push ~2h; stats + figures
~2.5h. +2h write-up.
