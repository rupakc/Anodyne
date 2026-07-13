# Human-in-the-Loop & Annotation

Automation is powerful, but sometimes you want a **person to have the final say**. Human-in-the-Loop
keeps people in control of the process, and Annotation lets your team record their judgments on the
data.

## What you can do today

- **Optional review gate.** Turn on review for a generation job and, instead of finishing
  automatically, it **pauses and waits for a person to approve or reject** it. A **review task**
  appears in a queue for someone on your team to act on. Leave the gate off and everything runs
  hands-free as before.
- **Annotations.** Add labels and notes to a dataset version — for example, flagging good or
  problematic records, or marking a version as "approved for use." These annotations stay attached
  to the version.
- **Feedback.** Capture free-form feedback about a dataset so lessons aren't lost.

All of this happens in the [Web UI](Web-UI), in a dedicated review-and-annotation area.

## Under the hood (in plain terms)

- The review gate works because generation runs on a **workflow engine that can pause** mid-job and
  resume once a human decides — no data is finalized until it's approved.
- Reviews, annotations, and feedback are all **kept per company**, so one organization never sees
  another's — see [Multi-Tenancy & Security](Multi-Tenancy-and-Security).
- Human review pairs naturally with the [Evaluation Engine](Evaluation-Engine): let the AI panel
  score a dataset, then have a person make the final call.
