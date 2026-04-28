# User Steering Instructions for singlify orchestrator

The orchestrator reads this file at Phase 0 (start of every cycle) via code-scout.

- Add instructions with the tag `[PENDING]` and the orchestrator will act on them.
- The orchestrator changes the tag to `[DONE cycle N]` when complete.
- This is the correct way to steer the agent between cycles.

---

## Format

```
[PENDING] Your instruction here.
```

## Active Instructions

<!-- Add your instructions below this line -->
