#!/usr/bin/env python3
"""Daily Reel pipeline, run by GitHub Actions (.github/workflows/daily-reel.yml).

Steps: check how past Reels performed -> Gemini writes the script -> Kaggle renders it
with reel_maker.py -> the MP4 is hosted briefly on GitHub -> Instagram publishes it.

Secrets come from the environment and are never printed:
  GEMINI_API_KEY, KAGGLE_API_TOKEN, IG_USER_ID, IG_TOKEN, GH_TOKEN,
  SECRETS_PAT (optional: lets the run store the refreshed Instagram token).
"""

import argparse
import base64
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "pipeline", "config.json")))
STATE_PATH = os.path.join(ROOT, "state", "history.json")
OUT_DIR = os.path.join(ROOT, "out")
IG = "https://graph.instagram.com/v25.0"
MARKER = 'JOB_B64 = "__JOB_B64__"'
REPO = os.environ.get("GITHUB_REPOSITORY", "")
RELEASE_TAG = "video-host"
TMP_BRANCH = "tmp-video"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "narration": {"type": "STRING"},
        "scenes": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": {"prompt": {"type": "STRING"}}, "required": ["prompt"]}},
        "caption": {"type": "STRING"},
        "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "propertyOrdering": ["title", "narration", "scenes", "caption", "hashtags"],
    "required": ["title", "narration", "scenes", "caption", "hashtags"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now():
    return dt.datetime.now(dt.timezone.utc)


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, check=check, capture_output=True, text=True, **kw)


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"posts": [], "pending": None}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


# --- Instagram ----------------------------------------------------------------
def ig(method, path, token, **params):
    """Call the Instagram API. Errors never include the URL, which carries the token."""
    params["access_token"] = token
    url = path if path.startswith("http") else f"{IG}/{path}"
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=60)
        else:
            r = requests.post(url, data=params, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Instagram {method} {path}: {type(e).__name__}") from None
    if r.status_code >= 400:
        raise RuntimeError(f"Instagram {method} {path}: HTTP {r.status_code} {r.text[:500]}")
    return r.json()


def refresh_token(token):
    """Long-lived tokens last 60 days; extend on every run and store the new one."""
    try:
        new = ig("GET", "https://graph.instagram.com/refresh_access_token", token,
                 grant_type="ig_refresh_token").get("access_token")
    except RuntimeError as e:
        log(f"Token refresh skipped ({e})")
        return token
    if not new:
        return token
    pat = os.environ.get("SECRETS_PAT")
    if pat and REPO:
        r = subprocess.run(["gh", "secret", "set", "IG_TOKEN", "--repo", REPO],
                           input=new, text=True, capture_output=True,
                           env={**os.environ, "GH_TOKEN": pat})
        log("Instagram token refreshed and stored" if r.returncode == 0
            else f"Could not store the refreshed token: {r.stderr.strip()[:200]}")
    else:
        log("Instagram token refreshed but not stored (no SECRETS_PAT secret)")
    return new


def collect_metrics(state, token):
    """Record how each Reel did: once after 48 hours, and a final time after 7 days."""
    for post in state["posts"]:
        if not post.get("media_id"):
            continue
        age_h = (now() - dt.datetime.fromisoformat(post["posted_at"])).total_seconds() / 3600
        stage = "7d" if age_h >= 168 else "48h" if age_h >= 48 else None
        if not stage or post.get("metrics_stage") == stage or post.get("metrics_stage") == "7d":
            continue
        try:
            data = ig("GET", f"{post['media_id']}/insights", token,
                      metric="views,reach,likes,comments,saved,shares,ig_reels_avg_watch_time")["data"]
            metrics = {m["name"]: m["values"][0]["value"] for m in data}
        except RuntimeError as e:
            log(f"Insights unavailable for {post['id']} ({str(e)[:160]}); using likes and comments")
            try:
                d = ig("GET", post["media_id"], token, fields="like_count,comments_count")
                metrics = {"likes": d.get("like_count", 0), "comments": d.get("comments_count", 0)}
            except RuntimeError as e2:
                log(f"No metrics for {post['id']}: {str(e2)[:160]}")
                continue
        post["metrics"], post["metrics_stage"] = metrics, stage
        log(f"Metrics ({stage}) for '{post['title']}': {metrics}")


def score(post):
    m = post.get("metrics") or {}
    engagement = m.get("likes", 0) + 2 * m.get("comments", 0) + 3 * m.get("saved", 0) + 3 * m.get("shares", 0)
    return m.get("views", 0) + 10 * engagement


def feedback_text(state):
    """Tell the writer which past Reels did best and worst, once there is enough data."""
    rated = [p for p in state["posts"] if p.get("metrics")]
    if len(rated) < CFG["min_posts_for_feedback"]:
        return ""
    rated.sort(key=score, reverse=True)

    def line(p):
        return f"- {p['title']} | hook: {p['hook']} | {json.dumps(p['metrics'])}"

    return ("\n\nHow our past Reels performed with our audience (learn from this: lean towards the "
            "subjects, hook style and tone of the best ones and away from the worst, but never "
            "repeat a topic).\nBest:\n" + "\n".join(line(p) for p in rated[:3])
            + "\nWorst:\n" + "\n".join(line(p) for p in rated[-3:]))


# --- script (Gemini) ----------------------------------------------------------
IDEAS_SCHEMA = {
    "type": "OBJECT",
    "properties": {"facts": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "fact": {"type": "STRING"},
            "why": {"type": "STRING"},
            "familiarity": {"type": "INTEGER"},
            "certainty": {"type": "INTEGER"},
            "filmable": {"type": "INTEGER"},
        },
        "required": ["fact", "why", "familiarity", "certainty", "filmable"]}}},
    "required": ["facts"],
}


def gemini(system, user, schema, check):
    """Ask Gemini for JSON. check(reply) returns an error string to retry, or None."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 1, "responseMimeType": "application/json",
                             "responseSchema": schema},
    }
    # Free Gemini models are often briefly overloaded: keep trying for ~25 minutes.
    # The backup model writes worse scripts, so it only gets every fourth attempt.
    main, backup = CFG["gemini_models"][0], CFG["gemini_models"][-1]
    last = "no attempt"
    for attempt in range(14):
        if attempt:
            time.sleep(min(30 * attempt, 150))
        model = backup if attempt % 4 == 3 else main
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, json=body, timeout=180,
                              headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        except requests.RequestException as e:
            last = type(e).__name__
            continue
        if r.status_code != 200:
            last = f"{model}: HTTP {r.status_code} {r.text[:300]}"
            log(f"Gemini attempt {attempt + 1} failed ({model}: HTTP {r.status_code})")
            continue
        try:
            reply = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, ValueError) as e:
            last = f"unreadable reply ({type(e).__name__})"
            continue
        last = check(reply)
        if last is None:
            log(f"Written by {model}")
            return reply
        log(f"Gemini attempt {attempt + 1} rejected: {last}")
    raise RuntimeError(f"Gemini failed: {last}")


def pick_fact(state):
    """Step 1: brainstorm many facts, have each rated, and keep the least-known sure one."""
    system = open(os.path.join(ROOT, "pipeline", "ideas_prompt.txt")).read()
    recent = [p["title"] for p in state["posts"][-CFG["remember_topics"]:]]
    user = (f"Topic area: {CFG['niche']}\n"
            f"Already used, do not repeat: {'; '.join(recent) or 'none'}" + feedback_text(state))

    def good(f):
        return f["familiarity"] <= 3 and f["certainty"] >= 9 and f["filmable"] >= 7

    reply = gemini(system, user, IDEAS_SCHEMA,
                   lambda r: None if any(good(f) for f in r["facts"]) else "no fact passed the ratings")
    facts = sorted(reply["facts"], key=lambda f: (f["familiarity"], -f["filmable"]))
    for f in facts:
        log(f"  known {f['familiarity']}/10, sure {f['certainty']}/10, filmable {f['filmable']}/10: {f['fact']}")
    return next(f for f in facts if good(f))


def write_script(state):
    fact = pick_fact(state)
    log(f"Chosen fact: {fact['fact']}")
    system = open(os.path.join(ROOT, "pipeline", "system_prompt.txt")).read()
    user = (f"The fact: {fact['fact']}\nWhy it is true: {fact['why']}\n"
            f"Number of scenes: {CFG['scenes']}")

    def check(plan):
        words = len(plan["narration"].split())
        if len(plan["scenes"]) < CFG["scenes"] or not 40 <= words <= 95:
            return f"bad plan: {len(plan['scenes'])} scenes, {words} words"
        if len(plan["caption"].strip()) < 150 or len(plan["hashtags"]) < 3:
            return "caption or hashtags too short"
        return None

    return gemini(system, user, SCHEMA, check)


# --- render (Kaggle) ----------------------------------------------------------
def kaggle(*args, check=True):
    return sh(["kaggle", *args], check=check)


def render(job):
    slug = f"{CFG['kaggle_user']}/{CFG['kernel_slug']}"
    script = open(os.path.join(ROOT, "reel_maker.py")).read()
    if MARKER not in script:
        raise RuntimeError("reel_maker.py is missing the JOB_B64 placeholder")
    b64 = base64.b64encode(json.dumps(job).encode()).decode()
    kdir = os.path.join(ROOT, "kernel-tmp")
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "reel_maker.py"), "w") as f:
        f.write(script.replace(MARKER, f'JOB_B64 = "{b64}"'))
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump({"id": slug, "title": CFG["kernel_slug"], "code_file": "reel_maker.py",
                   "language": "python", "kernel_type": "script", "is_private": True,
                   "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
                   "competition_sources": [], "kernel_sources": [], "model_sources": []}, f)
    r = kaggle("kernels", "push", "-p", kdir, "--accelerator", "NvidiaTeslaT4", check=False)
    log(f"Kaggle push: {(r.stdout + r.stderr).strip()[-200:]}")
    if r.returncode != 0 or "successfully pushed" not in r.stdout:
        raise RuntimeError("Kaggle push failed")

    time.sleep(90)
    deadline = time.time() + CFG["render_timeout_minutes"] * 60
    last = ""
    while time.time() < deadline:
        status = kaggle("kernels", "status", slug, check=False).stdout.strip().upper()
        if status != last:
            log(f"Kaggle: {status[-60:]}")
            last = status
        if "COMPLETE" in status:
            return
        if "ERROR" in status or "CANCEL" in status:
            fetch_output()
            for path in glob.glob(os.path.join(OUT_DIR, "*.log")):
                try:
                    lines = [e["data"].rstrip() for e in json.load(open(path))]
                except ValueError:
                    lines = open(path).read().splitlines()
                print("\n".join(lines[-40:]))
            raise RuntimeError("Kaggle render failed")
        time.sleep(120)
    raise RuntimeError("Kaggle render timed out")


def fetch_output(job_id=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    kaggle("kernels", "output", f"{CFG['kaggle_user']}/{CFG['kernel_slug']}", "-p", OUT_DIR, check=False)
    pattern = f"reel-{job_id}.mp4" if job_id else "reel-*.mp4"
    found = glob.glob(os.path.join(OUT_DIR, pattern))
    return found[0] if found else None


# --- hosting: Instagram downloads the video from a public URL -------------------
def gh(*args, check=True, **kw):
    return sh(["gh", *args], check=check, **kw)


def host_release(path):
    if gh("release", "view", RELEASE_TAG, "--repo", REPO, check=False).returncode != 0:
        gh("release", "create", RELEASE_TAG, "--repo", REPO, "--latest=false",
           "--title", "Temporary video hosting",
           "--notes", "Each Reel sits here for a few minutes while Instagram fetches it.")
    gh("release", "upload", RELEASE_TAG, path, "--repo", REPO, "--clobber")
    return f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{os.path.basename(path)}"


def unhost_release(path):
    gh("release", "delete-asset", RELEASE_TAG, os.path.basename(path), "--repo", REPO, "-y", check=False)


def host_branch(path):
    unhost_branch(path)
    sha = gh("api", f"repos/{REPO}/git/ref/heads/main", "--jq", ".object.sha").stdout.strip()
    gh("api", "-X", "POST", f"repos/{REPO}/git/refs", "-f", f"ref=refs/heads/{TMP_BRANCH}", "-f", f"sha={sha}")
    name = os.path.basename(path)
    body = json.dumps({"message": f"Temporary video {name}", "branch": TMP_BRANCH,
                       "content": base64.b64encode(open(path, "rb").read()).decode()})
    gh("api", "-X", "PUT", f"repos/{REPO}/contents/{name}", "--input", "-", input=body)
    time.sleep(20)
    return f"https://raw.githubusercontent.com/{REPO}/{TMP_BRANCH}/{name}"


def unhost_branch(path):
    gh("api", "-X", "DELETE", f"repos/{REPO}/git/refs/heads/{TMP_BRANCH}", check=False)


def publish(path, caption, token):
    user = os.environ["IG_USER_ID"]
    last = "no host tried"
    for host, unhost in ((host_release, unhost_release), (host_branch, unhost_branch)):
        try:
            url = host(path)
            log(f"Video hosted via {host.__name__}")
            container = ig("POST", f"{user}/media", token, media_type="REELS",
                           video_url=url, caption=caption)["id"]
            for _ in range(40):
                time.sleep(30)
                st = ig("GET", container, token, fields="status_code,status")
                if st.get("status_code") in ("FINISHED", "ERROR", "EXPIRED"):
                    break
            if st.get("status_code") != "FINISHED":
                last = f"{host.__name__}: Instagram said {st.get('status_code')} {st.get('status', '')}"
                log(last)
                continue
            media_id = ig("POST", f"{user}/media_publish", token, creation_id=container)["id"]
            try:
                link = ig("GET", media_id, token, fields="permalink").get("permalink", "")
            except RuntimeError:
                link = ""
            return media_id, link
        except (RuntimeError, subprocess.CalledProcessError) as e:
            last = f"{host.__name__}: {str(e)[:300]}"
            log(last)
        finally:
            unhost(path)
    raise RuntimeError(f"Instagram did not accept the video ({last})")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="post to Instagram (otherwise render only)")
    ap.add_argument("--reuse-render", action="store_true",
                    help="skip writing and rendering; use the Reel from the last run")
    ap.add_argument("--script-only", action="store_true", help="write the script and stop")
    ap.add_argument("--list-models", action="store_true", help="print the Gemini models this key can use")
    args = ap.parse_args()

    if args.list_models:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60)
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/")[-1])
        return

    state = load_state()
    token = os.environ.get("IG_TOKEN", "")
    if args.publish and not (token and os.environ.get("IG_USER_ID")):
        raise SystemExit("IG_TOKEN and IG_USER_ID secrets are needed to publish")
    if token:
        if args.publish:
            token = refresh_token(token)
        collect_metrics(state, token)
        save_state(state)

    if args.reuse_render:
        pending = state.get("pending")
        if not pending:
            raise SystemExit("Nothing to reuse: no rendered Reel is waiting")
        video = fetch_output(pending["id"])
        if not video:
            raise SystemExit(f"Kaggle no longer has reel-{pending['id']}.mp4")
    else:
        plan = write_script(state)
        job_id = now().strftime("%Y%m%d%H%M")
        tags = " ".join("#" + str(t).lstrip("#").replace(" ", "") for t in plan["hashtags"])
        pending = {
            "id": job_id,
            "title": plan["title"],
            "hook": plan["narration"].split(". ")[0][:120],
            "narration": plan["narration"],
            "caption": f"{plan['caption']}\n\n{tags} #aigenerated",
        }
        log(f"Topic: {plan['title']}")
        log(f"Narration: {plan['narration']}")
        log(f"Caption:\n{pending['caption']}")
        for i, s in enumerate(plan["scenes"][:CFG["scenes"]], 1):
            log(f"Scene {i}: {s['prompt']}")
        if args.script_only:
            return
        state["pending"] = pending
        save_state(state)
        render({"id": job_id, "narration": plan["narration"],
                "scenes": [{"prompt": s["prompt"]} for s in plan["scenes"][:CFG["scenes"]]],
                "voice": CFG["voice"], "quality": CFG["quality"]})
        video = fetch_output(job_id)
        if not video:
            raise RuntimeError("Kaggle finished but produced no Reel file")
    log(f"Reel ready: {os.path.basename(video)} ({os.path.getsize(video) / 1e6:.1f} MB)")

    if not args.publish:
        log("Render-only run: not posting. Download the video from this run's artifacts.")
        return

    media_id, link = publish(video, pending["caption"], token)
    log(f"Published: {link or media_id}")
    state["posts"].append({**pending, "media_id": media_id, "permalink": link,
                           "posted_at": now().isoformat(timespec="seconds")})
    state["pending"] = None
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        log(f"FAILED: {e}")
        sys.exit(1)
