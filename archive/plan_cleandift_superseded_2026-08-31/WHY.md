# Why this was archived (2026-08-31)

`PLAN_CLEANDIFT.md` was replaced by an implementation plan. This is the previous 541-line
feasibility/design document, kept because it is the only record of some material.

**What was carried forward into the new `PLAN_CLEANDIFT.md`:**
- §1/§1.1 reference-implementation mechanics and hyperparameters -> Appendix A.
- §5 pitfalls P1-P10 -> Appendix B, annotated with current status.
- §4.2's design -> superseded; the three departures are stated explicitly in the new doc (the
  changed success gate, a learned 256-d embedding instead of a learned scalar timestep, and
  skipping the §7.9 K-draw precondition).

**What was NOT carried forward, and where it lives instead:**
- §7 (SWEEP RESULTS 2026-08-27) is measured data. Its tables are preserved in `CLAUDE.md`, section
  "THE TIMESTEP SWEEP (2026-08-27)", including the §7.3 headroom numbers.
- §3.1's SD<->FM schedule conversion is summarised in `CLAUDE.md` (SD t=0 -> FM t~28; DIFT
  t=261 -> FM t~420).
- §4.1's gate: it already passed; superseded by the 2026-08-30 high-t sweep in `GOALS.md`, which
  found the coupled arm peaks at t~500 and collapses by t=900 while decoupled saturates.

Read this only for the full original text. The new plan is the operative document.
