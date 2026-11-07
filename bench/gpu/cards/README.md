# Experiment cards (Phase 3)

One card per hypothesis, each with the same five parts: hypothesis, change, measure, falsifier and result. A card's numbers come from PROFILE.md (pilot items 15 and 16, replay job 803599) unless the card names another job. Kernel ideas are timed first by replay or per-kernel timing, and only an idea that passes there goes to a sequential A/B on the pilot (bench/gpu/seqab). Every adopted change passes the 9-sample identity gate, except E2, which ships only as a named mode and is judged by the truth gate.

| card | hypothesis | status |
|---|---|---|
| [E3](E3-handoff-bypass.md) | the host work left in STAR's window can be cut by a fifth of a run's CPU | measuring: needs a per-thread CPU sampler (nsys sampling is off on Clipper) |
| [E2](E2-heavy-read-pruning.md) | capping the window search for heavy reads gains 1.3x in the kernels and passes the truth gate | gate settled; next the replay window-cap flavor |
| [SL](SL-more-slots.md) | need-based admission and the shared index make 3 slots pay for deep samples | first point queued |
| [XC](XC-cross-sample-cache.md) | keeping per-genome state resident across runs cuts device hold by 15% | design |
