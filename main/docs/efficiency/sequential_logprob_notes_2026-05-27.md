Short summary
Experiments were mostly fine; the FA2 speed comparison was misleading, not broken.

What went wrong operationally
Accidental full GRPO (jg92ywy3): meant to be a 10-step smoke; Modal didn’t get no_resume, so it resumed and trained ~160–182 on the shared GRPO checkpoint → wrote step_169/179.pt. Stopped. W&B fixed by retagging as smoke (doesn’t delete points; filter smoke out of prod charts).
Real smokes succeeded: minority (q6m0tmiu, 10 steps) and GRPO FA2 (kgmedwfu, 10 steps, no_resume + fresh wandb).
FA2 “slower” — was the experiment bad?
No major experiment bug — same workload (~850 tok/completion, same token_budget).
Wrong expectation — docs assumed FA2 speeds up HF train; we actually do ~100+ batch-1 logprob forwards per step, which is a bad shape for FA2.
FA2 is wired correctly in build_hf; the trainer loop (per-seq _completion_logprobs_hf) is why you don’t see a speedup (SDPA on H200 can look as good or better at B=1).
pcas3emd vs jg92ywy3 at “step 160” is an OK rough A/B (SDPA vs FA2), not a perfect lab test (different runs/containers).
What to do
Prod GRPO: don’t auto-resume from 169/179; use 159 or Anastasia’s line if you care about clean weights.
FA2: safe to keep for memory; don’t expect faster steps until logprob forwards are batched.
Charts: exclude smoke / run jg92ywy3 from production panels.
One line: Smokes validated the stack; the rogue run contaminated checkpoints and W&B clutter, not the science of FA2 — FA2 isn’t “broken,” it’s just not helping this sequential logprob path.