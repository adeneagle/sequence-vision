# Spherical-harmonics descriptors — removed 2026-08-26

Removed by direction. Rationale: the SH power spectrum has **no demonstrated use on a
downstream cryo-EM or cryo-ET task** relative to learned features. Its wins in this
project were all on the homolog *diagnostic*, a variance-decomposition proxy that
measures how smoothly a descriptor tracks sequence similarity — not informativeness and
not a task. A 35-dim hand-crafted summary can win that metric by being an easier
regression target. The external benchmark that did test this family on cryo-EM density
keypoints (CryoAlign, Nat. Commun. 15:1593) ranked spatial-distribution histogram
descriptors last, 2.5-9x behind learned/LRF-based SHOT.

This finishes a removal begun 2026-08-22, when `sh` was dropped from the homolog
diagnostic defaults for the same reason.

Files: `sh_invariants.py` (power-spectrum descriptor), `sh_canonical.py` (per-residue
local frames from SH; a documented failure -- 14.5-27.9 deg axis error).

`teachers/whole_map_canonical.py` was NOT removed: it used `_real_sh`/`fibonacci_sphere`
only as an internal scoring functional to break a sign ambiguity, never as a descriptor.
Those two helpers are now inlined there.
