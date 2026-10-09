#!/usr/bin/env python3
"""Weekly status report, run by GitHub Actions every Sunday.

Opens a GitHub issue with the week's posts, their numbers and the follower count.
GitHub emails that issue to the repository owner. The issue carries a Pause/Resume
button: it opens a pre-filled issue, and control.yml acts on it once it is created.
"""

import datetime as dt
import json
import os
import subprocess
from urllib.parse import quote

import post_reel as p

REPO = os.environ["GITHUB_REPOSITORY"]
WORKFLOW = "daily-reel.yml"


def gh_json(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def button(action, label, colour):
    body = "Just press the green button below. Nothing else is needed."
    url = f"https://github.com/{REPO}/issues/new?title={action}&body={quote(body)}"
    badge = f"https://img.shields.io/badge/{quote(label)}-{colour}?style=for-the-badge"
    return f"[![{label}]({badge})]({url})\n\n[{label}]({url}) (opens GitHub, then press the green button)"


def main():
    state = p.load_state()
    token = os.environ.get("IG_TOKEN", "")
    user = os.environ.get("IG_USER_ID", "")
    today = p.now()
    week_ago = today - dt.timedelta(days=7)
    lines = [f"## Weekly Reel report, {today.strftime('%d %b %Y')}", ""]

    # --- is the automation on?
    wf = gh_json("api", f"repos/{REPO}/actions/workflows/{WORKFLOW}") or {}
    auto = os.environ.get("AUTO_POST") == "true"
    paused = wf.get("state", "active") != "active"
    if paused:
        lines += ["**Status: paused.** No Reels are being made or posted.", "",
                  button("resume", "Resume the automation", "2ea44f")]
    elif not auto:
        lines += ["**Status: not switched on yet.** The daily schedule is waiting to be turned on."]
    else:
        lines += ["**Status: running.** Three Reels are posted every day.", "",
                  button("pause", "Pause the automation", "d73a49")]
    lines.append("")

    # --- followers
    if token and user:
        try:
            me = p.ig("GET", user, token, fields="username,followers_count,media_count")
            count = me.get("followers_count")
            history = state.setdefault("followers", [])
            previous = history[-1]["count"] if history else None
            history.append({"date": today.date().isoformat(), "count": count})
            change = "" if previous is None else f" ({count - previous:+d} since the last report)"
            lines += [f"**Followers:** {count}{change}  ",
                      f"**Account:** @{me.get('username', '?')}, {me.get('media_count', '?')} posts in total", ""]
        except RuntimeError as e:
            lines += [f"**Followers:** could not be read ({str(e)[:120]})", ""]
    else:
        lines += ["**Instagram is not connected yet**, so there are no follower or post numbers.", ""]

    # --- this week's posts, with fresh numbers
    week = [x for x in state["posts"]
            if dt.datetime.fromisoformat(x["posted_at"]) >= week_ago]
    lines += [f"### Reels posted this week: {len(week)}", ""]
    if week:
        lines += ["| Day | Reel | Views | Likes | Comments | Saves | Shares |", "|---|---|---|---|---|---|---|"]
        for post in week:
            m = post.get("metrics") or {}
            if token and post.get("media_id"):
                try:
                    data = p.ig("GET", f"{post['media_id']}/insights", token,
                                metric="views,reach,likes,comments,saved,shares")["data"]
                    m = {x["name"]: x["values"][0]["value"] for x in data}
                except RuntimeError:
                    try:
                        d = p.ig("GET", post["media_id"], token, fields="like_count,comments_count")
                        m = {"likes": d.get("like_count"), "comments": d.get("comments_count")}
                    except RuntimeError:
                        pass
            post["week_metrics"] = m
            day = dt.datetime.fromisoformat(post["posted_at"]).strftime("%a %d %b")
            title = f"[{post['title']}]({post['permalink']})" if post.get("permalink") else post["title"]
            cells = [str(m.get(k, "–")) for k in ("views", "likes", "comments", "saved", "shares")]
            lines.append(f"| {day} | {title} | " + " | ".join(cells) + " |")
        rated = [x for x in week if (x.get("week_metrics") or {}).get("views") is not None]
        if rated:
            best = max(rated, key=lambda x: x["week_metrics"]["views"])
            lines += ["", f"**Best this week:** {best['title']} ({best['week_metrics']['views']} views)"]
        for post in week:
            post.pop("week_metrics", None)
    lines.append("")

    # --- runs that failed
    runs = gh_json("run", "list", "--repo", REPO, "--workflow", WORKFLOW, "--limit", "40",
                   "--json", "conclusion,createdAt,event") or []
    recent = [r for r in runs if dt.datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) >= week_ago
              and r["event"] == "schedule"]
    failed = sum(r["conclusion"] == "failure" for r in recent)
    lines += [f"**Daily runs this week:** {len(recent)}, of which {failed} failed.",
              f"[See the run history](https://github.com/{REPO}/actions/workflows/{WORKFLOW})"]

    p.save_state(state)
    # Mentioning and assigning the owner makes GitHub email them even if they
    # are not watching the repository.
    owner = REPO.split("/")[0]
    body = "\n".join(lines) + f"\n\ncc @{owner}"
    print(body)
    subprocess.run(["gh", "issue", "create", "--repo", REPO, "--assignee", owner,
                    "--title", f"Weekly Reel report, {today.strftime('%d %b %Y')}",
                    "--body", body], check=True)


if __name__ == "__main__":
    main()
