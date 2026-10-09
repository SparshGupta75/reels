#!/usr/bin/env python3
"""Merge this run's history into the latest history on GitHub without losing entries.

Usage: merge_state.py <this run's history.json> <latest history.json, updated in place>
Posts are matched by id; a post that only one side knows about is always kept.
"""

import json
import sys

mine = json.load(open(sys.argv[1]))
latest = json.load(open(sys.argv[2]))

posts = {p["id"]: p for p in latest.get("posts", [])}
posts.update({p["id"]: p for p in mine.get("posts", [])})        # this run's copy is the newer one
merged = dict(latest)
merged.update(mine)
merged["posts"] = sorted(posts.values(), key=lambda p: p.get("posted_at", ""))
followers = {f["date"]: f for f in latest.get("followers", []) + mine.get("followers", [])}
if followers:
    merged["followers"] = [followers[d] for d in sorted(followers)]

with open(sys.argv[2], "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
    f.write("\n")
