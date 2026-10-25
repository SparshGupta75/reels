#!/usr/bin/env python3
"""Publish a trial Reel that has already been rendered and reviewed.

Usage: publish_trial.py pipeline/trials/<name>.json out/reel-<id>.mp4 out/caption-<id>.txt

The video is the exact file from the review artifact, not a re-render, so what gets
posted is what was approved. The post is added to state/history.json so metrics are
collected and the daily cap counts it.
"""

import glob
import json
import os
import sys

import post_reel as p


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: publish_trial.py <trial.json> <video.mp4> <caption.txt>")
    trial = json.load(open(sys.argv[1]))
    video, caption_file = sys.argv[2], sys.argv[3]
    caption = open(caption_file).read().strip()
    job = trial["job"]

    token = os.environ.get("IG_TOKEN", "")
    if not (token and os.environ.get("IG_USER_ID")):
        raise SystemExit("IG_TOKEN and IG_USER_ID are needed to publish")
    token = p.refresh_token(token)

    covers = sorted(glob.glob(os.path.join(os.path.dirname(video), "cover-*.jpg")))
    p.log(f"Publishing: {trial['title']} [{trial['series']}]")
    p.log(f"Video: {video} ({os.path.getsize(video) / 1e6:.1f} MB), cover: {covers[0] if covers else 'none'}")
    media_id, link = p.publish(video, caption, token, covers[0] if covers else None)
    p.log(f"Published: {link or media_id} (id {media_id})")

    state = p.load_state()
    state["posts"].append({
        "id": job["id"],
        "title": trial["title"],
        "series": trial["series"],
        "hook": job["narration"].split(". ")[0][:120],
        "fact": trial.get("fact") or trial["title"],
        "narration": job["narration"],
        "caption": caption,
        "hook_text": job["hook_text"],
        "cover_text": job["cover_text"],
        "media_id": media_id,
        "permalink": link,
        "posted_at": p.now().isoformat(timespec="seconds"),
    })
    state["pending"] = None
    p.save_state(state)
    p.log(f"History updated: {len(state['posts'])} posts")


if __name__ == "__main__":
    main()
