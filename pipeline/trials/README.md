# Trials

Hand-written Reels used to test a format before it joins the daily pipeline. Render them with
the **Trial Reel** workflow, review the artifact, then publish the exact file with
**Publish Trial Reel** (no re-render).

| File | Series | Status |
|---|---|---|
| `airplane-bathroom.json` | Why Is There No... | Published 2026-10-25 — https://www.instagram.com/reel/De6Q0bQPcGk/ |
| `jacket-button.json` | The Hidden Feature | Rendered, 15.2s. Visual caveat: the spare button is described but the stills show the jacket's front button. |
| `jeans-pocket.json` | The Hidden Feature | **Not renderable with AI stills.** Three attempts; the model draws back pockets instead of the fifth pocket every time. Needs one real macro clip. |

## What the trials proved

The image model draws the *object* reliably and the *detail* unreliably. A detail that is
large, central and in the model's vocabulary (a window row, a fuselage) survives. A detail
that is small, hidden or named after something the model has not seen (fifth pocket, coin
pocket, spare button inside a lining) gets replaced by the generic object.

So: choose Hidden Feature and Label Decoder facts where the detail itself is big in frame,
or plan on one real clip per Reel.
