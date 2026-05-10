---
name: Don't run long commands — provide them for Alexis to run
description: Long-running scripts (retrain, audit, sleep, etc.) belong in Alexis's terminal, not Bash tool calls — even when auto mode is active
type: feedback
---

For long-running scripts (retrain, audit pathology, sleep, probe, anything >~1 minute), provide the command for Alexis to run in his Windows Terminal rather than executing via the Bash tool.

**Why:** Alexis owns the operational loop — he's watching the terminal, has the right env, and can interrupt or course-correct mid-run. When I run multi-minute commands via Bash, he loses observability and the conversation fills with progress logs that are noise to him. The handoff prompt for this project specifically called this out: "Alexis runs long things in his Windows Terminal. Always provide the command." Auto mode's "execute immediately" framing doesn't override an established operational convention.

**How to apply:** Smoke tests and quick verifications (~seconds) are fine via Bash. For anything that takes minutes — retrain wrappers, audit scripts, sleep cycles, probe runs, batch judges, end-to-end QA pipelines — write the command into the response and stop. If I think a run is needed for the task to finish, surface it as "next step: run X" rather than launching it. This applies even after the user has consented to a plan that includes the run; consent to the plan ≠ consent to me executing the run.
