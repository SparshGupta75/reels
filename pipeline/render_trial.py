#!/usr/bin/env python3
"""Render a hand-written trial Reel and keep the files for review.

Run from the repository root:  python pipeline/render_trial.py pipeline/trials/jeans-pocket.json
It renders on Kaggle under its own kernel slug so it never touches the daily pipeline,
and it never publishes anything to Instagram.
"""

import json
import os
import sys

import post_reel as p


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: render_trial.py pipeline/trials/<name>.json")
    trial = json.load(open(sys.argv[1]))
    job = dict(trial["job"])
    job.setdefault("voice", p.CFG["voice"])
    job.setdefault("quality", p.CFG["quality"])
    job.setdefault("visual", "stills")
    if len(job.get("scenes", [])) < 4:
        raise SystemExit("a trial Reel needs at least 4 scenes")

    slug = f"{p.CFG['kaggle_user']}/{trial['slug']}"
    p.log(f"Trial: {trial['title']} [{trial['series']}]")
    p.log(f"Kernel: {slug}")
    p.log(f"Narration: {job['narration']}")
    for i, s in enumerate(job["scenes"], 1):
        p.log(f"Scene {i}: {s['prompt']}")

    p.render(job, slug=slug)
    video = p.fetch_output(job["id"], slug=slug)
    if not video:
        raise RuntimeError("Kaggle finished but produced no Reel file")

    caption = os.path.join(p.OUT_DIR, f"caption-{job['id']}.txt")
    with open(caption, "w") as f:
        f.write(trial["caption"].strip() + "\n")
    p.log(f"Ready: {os.path.basename(video)} ({os.path.getsize(video) / 1e6:.1f} MB)")
    p.log(f"Caption written: {caption}")


if __name__ == "__main__":
    main()
