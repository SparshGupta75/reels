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
        "hook_text": {"type": "STRING"},
        "cover_text": {"type": "STRING"},
    },
    "propertyOrdering": ["title", "narration", "scenes", "caption", "hashtags", "hook_text", "cover_text"],
    "required": ["title", "narration", "scenes", "caption", "hashtags", "hook_text", "cover_text"],
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
    # A post whose ID was not saved (the history save failed that day) is found again by its link.
    if any(not p.get("media_id") and p.get("permalink") for p in state["posts"]):
        try:
            media = ig("GET", f"{os.environ['IG_USER_ID']}/media", token, fields="id,permalink", limit=50)["data"]
            ids = {m.get("permalink", "").rstrip("/"): m["id"] for m in media}
            for p in state["posts"]:
                if not p.get("media_id") and p.get("permalink", "").rstrip("/") in ids:
                    p["media_id"] = ids[p["permalink"].rstrip("/")]
                    log(f"Recovered the Instagram ID for '{p['title']}'")
        except (RuntimeError, KeyError) as e:
            log(f"Could not look up missing post IDs: {str(e)[:120]}")
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
            "wow": {"type": "INTEGER"},
            "familiarity": {"type": "INTEGER"},
            "certainty": {"type": "INTEGER"},
            "filmable": {"type": "INTEGER"},
        },
        "required": ["fact", "why", "wow", "familiarity", "certainty", "filmable"]}}},
    "required": ["facts"],
}


def gemini(system, user, schema=None, check=None, search=False, rounds=6, weakest=True, backwards=False):
    """Ask Gemini. With a schema the reply is parsed JSON, otherwise plain text.
    check(reply) returns an error string to retry, or None. search=True lets it use Google."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 1 if schema else 0},
    }
    if schema:
        body["generationConfig"].update(responseMimeType="application/json", responseSchema=schema)
    if search:
        body["tools"] = [{"google_search": {}}]
    # Free models are often overloaded or out of free quota: walk down the list of
    # models (best first), and go round the list again after a pause, for ~25 minutes.
    last = "no attempt"
    for round_no in range(rounds):
        if round_no:
            time.sleep(min(60 * round_no, 240))
        # The last model in the list is the weakest; creative steps leave it out at first.
        models = CFG["gemini_models"] if weakest else CFG["gemini_models"][:-1]
        for model in reversed(models[:-1]) if backwards else models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                r = requests.post(url, json=body, timeout=180,
                                  headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            except requests.RequestException as e:
                last = f"{model}: {type(e).__name__}"
                continue
            if r.status_code != 200:
                last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                log(f"Gemini {model}: HTTP {r.status_code}")
                continue
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts)
                reply = json.loads(text) if schema else text.strip()
            except (KeyError, IndexError, ValueError) as e:
                last = f"{model}: unreadable reply ({type(e).__name__})"
                continue
            last = check(reply) if check else None
            if last is None:
                log(f"Answered by {model}")
                return reply
            log(f"Gemini {model} rejected: {last}")
    raise RuntimeError(f"Gemini failed: {last}")


def creative(system, user, schema, check):
    """For the steps where quality matters most: wait ~25 minutes for a strong model
    before settling for the weakest one."""
    try:
        return gemini(system, user, schema, check, weakest=False)
    except RuntimeError as e:
        log(f"Strong models unavailable ({str(e)[:80]}); using the backup model")
        return gemini(system, user, schema, check, rounds=2)


def verify(fact):
    """Second opinion on a fact.
    Returns the fact (possibly reworded to be precise) or None if it is not solid."""
    system = ("You are a strict fact checker. Check the claim against reliable sources. "
              "Line 1 of your reply must be exactly one word: TRUE, FALSE or UNSURE. Use TRUE only "
              "if the claim is correct as stated or with a small wording fix, and is well established. "
              "Line 2: one sentence on why. Line 3, only if TRUE: the claim restated accurately in one "
              "plain sentence, with the correct numbers.")
    user = f"Claim: {fact['fact']}\nExplanation given: {fact['why']}"

    def check(reply):
        return None if reply.split()[:1] and reply.split()[0].strip(".:*").upper() in ("TRUE", "FALSE", "UNSURE") \
            else "no verdict"

    # Google Search grounding is refused on the free tier (HTTP 429 on every model), so the
    # check runs without it unless "search_fact_check" is switched on in the config.
    reply = None
    if CFG.get("search_fact_check"):
        try:
            reply = gemini(system, user, check=check, search=True, rounds=1)
        except RuntimeError:
            log("Fact check with Google Search was refused; checking without it")
    if reply is None:
        reply = gemini(system, user, check=check)
    lines = [x.strip() for x in reply.splitlines() if x.strip()]
    verdict = lines[0].strip(".:*").upper()
    log(f"Fact check: {verdict}. {lines[1] if len(lines) > 1 else ''}")
    if verdict != "TRUE":
        return None
    checked = lines[2] if len(lines) > 2 else fact["fact"]

    # One checker has passed a claim that another rejected, so a second, sceptical one
    # (asked a different way, starting from a different model) must agree as well.
    sceptic = ("You debunk popular 'amazing facts'. Many are myths, exaggerations, or true only in "
               "rare cases. Decide whether this claim is literally true for the ordinary, typical case, "
               "exactly as worded. Line 1 of your reply must be exactly one word: TRUE, FALSE or UNSURE. "
               "Answer FALSE if any part is wrong or only true in unusual cases, and UNSURE if you cannot "
               "be certain. Line 2: one sentence on why.")
    second = gemini(sceptic, f"Claim: {checked}", check=check, backwards=True)
    verdict2 = second.split()[0].strip(".:*").upper()
    reason = " ".join(second.split()[1:])[:300]
    log(f"Second opinion: {verdict2}. {reason}")
    return checked if verdict2 == "TRUE" else None


STOPWORDS = set("""a an the of to in on at for from by with and or but is are was were be been being it its
this that these those as than then so not no can could will would do does did has have had they them their you
your we our he she his her up down out into over under about more most very only just also even when while if
because which who what how why when where there here actually really every each one two""".split())


def keywords(text):
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w.rstrip("s") for w in words if len(w) > 2 and w not in STOPWORDS}


def already_used(fact, used):
    """True when a fact shares most of its key words with one posted before."""
    new = keywords(fact)
    for old in used:
        seen = keywords(old)
        if new and seen and len(new & seen) / min(len(new), len(seen)) >= 0.5:
            log(f"Skipping a repeat of an earlier Reel: {fact}")
            return True
    return False


def pick_fact(state):
    """Step 1: brainstorm many facts, have each rated, then fact-check the best ones."""
    system = open(os.path.join(ROOT, "pipeline", "ideas_prompt.txt")).read()
    # Every fact ever used is remembered, not just recent ones.
    used = [p.get("fact") or p["title"] for p in state["posts"]]
    recent = used
    user = (f"Topic area: {CFG['niche']}\n"
            f"Already used, do not repeat: {'; '.join(recent) or 'none'}" + feedback_text(state))

    def good(f):
        return f["familiarity"] <= 3 and f["certainty"] >= 9 and f["filmable"] >= 8 and f["wow"] >= 8

    for _ in range(3):
        reply = creative(system, user, IDEAS_SCHEMA,
                         lambda r: None if any(good(f) for f in r["facts"]) else "no fact passed the ratings")
        facts = sorted(reply["facts"], key=lambda f: (-f["wow"], f["familiarity"]))
        for f in facts:
            log(f"  wow {f['wow']}, known {f['familiarity']}, sure {f['certainty']}, "
                f"filmable {f['filmable']}: {f['fact']}")
        fresh = [f for f in facts if good(f) and not already_used(f["fact"], used)]
        for f in fresh[:4]:
            log(f"Checking: {f['fact']}")
            checked = verify(f)
            if checked:
                return {**f, "fact": checked}
    raise RuntimeError("no fact survived the fact check")


def write_script(state):
    fact = pick_fact(state)
    log(f"Chosen fact: {fact['fact']}")
    system = open(os.path.join(ROOT, "pipeline", "system_prompt.txt")).read()
    user = (f"The fact: {fact['fact']}\nWhy it is true: {fact['why']}\n"
            f"Number of scenes: {CFG['scenes']}")

    def check(plan):
        words = len(plan["narration"].split())
        low, high = CFG["narration_words"]
        if len(plan["scenes"]) < CFG["scenes"] or not low <= words <= high:
            return f"bad plan: {len(plan['scenes'])} scenes, {words} words"
        jargon = [w for w in CFG["jargon_words"] if w in plan["narration"].lower()]
        if jargon:
            return f"narration uses science-class words: {', '.join(jargon)}"
        if len(plan["caption"].strip()) < 150 or len(plan["hashtags"]) < 3:
            return "caption or hashtags too short"
        caption = plan["caption"].lower()
        if caption.lstrip().startswith("did you know") or "\u2014" in caption or "follow for more" in caption:
            return "caption breaks the style rules"
        return None

    plan = creative(system, user, SCHEMA, check)

    # A second pass removes details the writer may have made up (names, places, years,
    # numbers) and absolute claims the checked fact does not support.
    editor = ("You are a strict fact-checking editor for short video scripts. You get a checked fact, "
              "then a narration and a caption written from it. Rewrite both so that every claim is "
              "supported by the checked fact or is something you are certain is true. Remove every "
              "person's name, institution and year that is not in the checked fact. Delete or soften "
              "any place or number you are not certain of, and any absolute "
              "claim (only, never, cannot, always) that the checked fact does not support. Keep the "
              "style, tone, structure, line breaks and length, and keep the sentences flowing naturally; "
              "change as little as possible.")
    edit_schema = {"type": "OBJECT", "properties": {"narration": {"type": "STRING"}, "caption": {"type": "STRING"}},
                   "required": ["narration", "caption"]}
    try:
        fixed = gemini(editor, f"Checked fact: {fact['fact']}\nWhy: {fact['why']}\n\n"
                               f"Narration: {plan['narration']}\n\nCaption: {plan['caption']}",
                       edit_schema, lambda r: check({**plan, **r}), rounds=2)
        plan.update(fixed)
    except RuntimeError as e:
        log(f"Editing pass skipped ({str(e)[:100]})")
    plan["fact"] = fact["fact"]
    return plan


# --- the other formats: quiz and three-things -----------------------------------
GREEN = "&H50D050&"
FORMATS = ["regular", "quiz", "three"]          # morning, afternoon, evening


def ist_slot(t):
    """Three Reels a day, India time: morning (before 14:00), afternoon (14:00-18:00), evening."""
    ist = t.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    return (ist.date(), 0 if ist.hour < 14 else 1 if ist.hour < 18 else 2)


def shared_rules():
    """The picture, caption and hashtag rules, taken from the main prompt so every format shares them."""
    text = open(os.path.join(ROOT, "pipeline", "system_prompt.txt")).read()
    pictures = text[text.index("Each scene prompt"):text.index("The caption must read")]
    caption = text[text.index("The caption must read"):text.index("hook_text is")]
    return ("\n\nRules for every picture description. " + pictures.replace("Each scene prompt describes", "Each one describes")
            + "\n" + caption)


def verify_many(claims, expected):
    """Two independent checks of several claims at once. Returns None if every verdict matches
    `expected` (a list of "TRUE"/"FALSE") in both checks, else a description of the mismatch."""
    schema = {"type": "OBJECT", "properties": {"verdicts": {"type": "ARRAY", "items": {"type": "STRING"}}},
              "required": ["verdicts"]}
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompts = [
        ("You are a strict fact checker. For each numbered claim answer TRUE only if it is correct as stated "
         "and well established, FALSE if it is wrong, UNSURE otherwise. Return one verdict per claim, in order.", False),
        ("You debunk popular 'amazing facts'. Many are myths, exaggerations, or true only in rare cases. For each "
         "numbered claim decide whether it is literally true for the ordinary, typical case exactly as worded: "
         "TRUE, FALSE (any part wrong, or a known myth) or UNSURE. Return one verdict per claim, in order.", True),
    ]
    for system, backwards in prompts:
        reply = gemini(system, listing, schema,
                       lambda r: None if len(r["verdicts"]) == len(claims) else "wrong number of verdicts",
                       backwards=backwards)
        got = [v.strip().upper() for v in reply["verdicts"]]
        log(f"Fact check: {got}")
        if got != expected:
            return f"check gave {got}, needed {expected}"
    return None


def plain(text):
    return [w for w in CFG["jargon_words"] if w in text.lower()]


def str_list(n):
    return {"type": "ARRAY", "items": {"type": "STRING"}}


def write_quiz(state, seed):
    import random

    used = [p.get("fact") or p["title"] for p in state["posts"]]
    system = open(os.path.join(ROOT, "pipeline", "quiz_prompt.txt")).read() + shared_rules()
    schema = {"type": "OBJECT", "properties": {
        "title": {"type": "STRING"},
        "statements": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING"}, "is_true": {"type": "BOOLEAN"}, "picture": {"type": "STRING"}},
            "required": ["text", "is_true", "picture"]}},
        "why_true": {"type": "STRING"}, "myths_corrected": {"type": "STRING"},
        "opening_picture": {"type": "STRING"}, "pause_picture": {"type": "STRING"},
        "reveal_pictures": str_list(2), "myth_pictures": str_list(2),
        "caption": {"type": "STRING"}, "hashtags": str_list(5)},
        "required": ["title", "statements", "why_true", "myths_corrected", "opening_picture", "pause_picture",
                     "reveal_pictures", "myth_pictures", "caption", "hashtags"]}
    user = (f"Topic area: {CFG['niche']}\nAlready used, do not repeat any of these facts or myths: "
            f"{'; '.join(used) or 'none'}")

    def check(q):
        st = q["statements"]
        if len(st) != 3 or sum(x["is_true"] for x in st) != 1:
            return "need exactly three statements with exactly one true"
        if any(len(x["text"].split()) > 14 for x in st):
            return "a statement is too long"
        if len(q["reveal_pictures"]) < 2 or len(q["myth_pictures"]) < 2:
            return "missing pictures"
        words = " ".join([x["text"] for x in st] + [q["why_true"], q["myths_corrected"]])
        if plain(words):
            return f"science-class words: {', '.join(plain(words))}"
        if len(q["caption"].strip()) < 150 or q["caption"].lower().lstrip().startswith("did you know"):
            return "caption breaks the rules"
        if any(already_used(x["text"], used) for x in st if x["is_true"]):
            return "the true fact was used before"
        return None

    for _ in range(3):
        q = creative(system, user, schema, check)
        true = next(x for x in q["statements"] if x["is_true"])
        myths = [x for x in q["statements"] if not x["is_true"]]
        for x in q["statements"]:
            log(f"  {'TRUE ' if x['is_true'] else 'MYTH '} {x['text']}")
        problem = verify_many([true["text"]] + [m["text"] for m in myths], ["TRUE", "FALSE", "FALSE"])
        if problem:
            log(f"Quiz rejected: {problem}")
            continue
        order = q["statements"][:]
        random.Random(seed).shuffle(order)                 # the true one lands on a different letter each day
        letter = "ABC"[order.index(true)]
        segments = [{"say": "Two of these are myths. Which one is true?", "header": "WHICH ONE IS *TRUE?*",
                     "pics": [q["opening_picture"]]}]
        for name, x in zip("ABC", order):
            segments.append({"say": f"{name}. {x['text']}", "header": f"*{name}*", "pics": [x["picture"]]})
        segments += [
            {"say": "Got your answer?", "header": "GOT YOUR *ANSWER?*", "pics": [q["pause_picture"]]},
            {"say": f"It's {letter}. {q['why_true']}", "header": f"*{letter} IS TRUE*", "header_colour": GREEN,
             "chime": True, "pics": q["reveal_pictures"][:2]},
            {"say": q["myths_corrected"], "split": [[q["myth_pictures"][0], "MYTH"], [q["myth_pictures"][1], "MYTH"]]},
        ]
        return {"title": q["title"], "caption": q["caption"], "hashtags": q["hashtags"], "segments": segments,
                "fact": f"{true['text']} (myths: {myths[0]['text']} / {myths[1]['text']})",
                "hook_text": "", "cover_text": "*TRUE* OR MYTH", "mood": "tense"}
    raise RuntimeError("no quiz passed the fact check")


def write_three(state):
    used = [p.get("fact") or p["title"] for p in state["posts"]]
    system = open(os.path.join(ROOT, "pipeline", "three_prompt.txt")).read() + shared_rules()
    schema = {"type": "OBJECT", "properties": {
        "title": {"type": "STRING"}, "subject": {"type": "STRING"},
        "mystery_picture": {"type": "STRING"}, "reveal_picture": {"type": "STRING"},
        "facts": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING"}, "pictures": str_list(2)}, "required": ["text", "pictures"]}},
        "comparison_fact": {"type": "INTEGER"},
        "top_picture": {"type": "STRING"}, "top_label": {"type": "STRING"},
        "bottom_picture": {"type": "STRING"}, "bottom_label": {"type": "STRING"},
        "closing_picture": {"type": "STRING"}, "cover_text": {"type": "STRING"},
        "caption": {"type": "STRING"}, "hashtags": str_list(5)},
        "required": ["title", "subject", "mystery_picture", "reveal_picture", "facts", "comparison_fact",
                     "top_picture", "top_label", "bottom_picture", "bottom_label", "closing_picture",
                     "cover_text", "caption", "hashtags"]}
    user = (f"Topic area: {CFG['niche']}\nSubjects and facts already used, choose a different subject: "
            f"{'; '.join(used) or 'none'}")

    def check(t):
        if len(t["facts"]) != 3 or any(len(f["pictures"]) < 1 for f in t["facts"]):
            return "need exactly three facts, each with pictures"
        if any(len(f["text"].split()) > 22 for f in t["facts"]):
            return "a fact is too long"
        words = " ".join(f["text"] for f in t["facts"])
        if plain(words):
            return f"science-class words: {', '.join(plain(words))}"
        if len(t["caption"].strip()) < 150 or t["caption"].lower().lstrip().startswith("did you know"):
            return "caption breaks the rules"
        if any(already_used(f["text"], used) for f in t["facts"]):
            return "one of the facts was used before"
        return None

    for _ in range(3):
        t = creative(system, user, schema, check)
        subject = t["subject"].strip().rstrip(".")
        name = subject.split(" ", 1)[-1]
        for f in t["facts"]:
            log(f"  {f['text']}")
        problem = verify_many([f"About {subject}: {f['text']}" for f in t["facts"]], ["TRUE"] * 3)
        if problem:
            log(f"Rejected: {problem}")
            continue
        segments = [
            {"say": "Can you tell what this is?", "header": "WHAT IS *THIS?*", "pics": [t["mystery_picture"]]},
            {"say": f"It's {subject}, and here are three things you didn't know about it.", "chime": True,
             "pics": [t["reveal_picture"]]},
        ]
        for n, (word, f) in enumerate(zip(("One", "Two", "Three"), t["facts"]), 1):
            seg = {"say": f"{word}. {f['text']}", "header": f"*{n}* / 3"}
            if t["comparison_fact"] == n and t["top_picture"].strip() and t["bottom_picture"].strip():
                seg["split"] = [[t["top_picture"], " ".join(t["top_label"].upper().split()[:4])],
                                [t["bottom_picture"], " ".join(t["bottom_label"].upper().split()[:4])]]
            else:
                seg["pics"] = f["pictures"][:2]
            segments.append(seg)
        segments.append({"say": "Which one surprised you most?", "pics": [t["closing_picture"]]})
        return {"title": t["title"], "caption": t["caption"], "hashtags": t["hashtags"], "segments": segments,
                "fact": f"{name}: " + " / ".join(f["text"] for f in t["facts"]),
                "hook_text": "", "cover_text": t["cover_text"], "mood": "curious"}
    raise RuntimeError("no three-things script passed the fact check")


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


def wait_container(container, token):
    """Wait for Instagram to finish processing an upload. Returns None when ready, else why not."""
    st = {}
    for _ in range(40):
        time.sleep(30)
        st = ig("GET", container, token, fields="status_code,status")
        if st.get("status_code") in ("FINISHED", "ERROR", "EXPIRED"):
            break
    return None if st.get("status_code") == "FINISHED" else f"{st.get('status_code')} {st.get('status', '')}"


def share_to_story(url, token):
    """Also put the Reel's video on the account's Story. Never allowed to fail the run."""
    user = os.environ["IG_USER_ID"]
    try:
        container = ig("POST", f"{user}/media", token, media_type="STORIES", video_url=url)["id"]
        problem = wait_container(container, token)
        if problem:
            log(f"Story not shared: Instagram said {problem}")
            return
        ig("POST", f"{user}/media_publish", token, creation_id=container)
        log("Shared to Story")
    except RuntimeError as e:
        log(f"Story not shared: {str(e)[:300]}")


def publish(path, caption, token, cover=None):
    user = os.environ["IG_USER_ID"]
    last = "no host tried"
    for host, unhost in ((host_release, unhost_release), (host_branch, unhost_branch)):
        hosted_cover = False
        try:
            url = host(path)
            log(f"Video hosted via {host.__name__}")
            extra = {}
            if cover and host is host_release:
                try:
                    extra["cover_url"] = host_release(cover)
                    hosted_cover = True
                except subprocess.CalledProcessError:
                    log("Cover image could not be hosted; posting without it")
            container = ig("POST", f"{user}/media", token, media_type="REELS",
                           video_url=url, caption=caption, **extra)["id"]
            problem = wait_container(container, token)
            if problem:
                last = f"{host.__name__}: Instagram said {problem}"
                log(last)
                continue
            media_id = ig("POST", f"{user}/media_publish", token, creation_id=container)["id"]
            try:
                link = ig("GET", media_id, token, fields="permalink").get("permalink", "")
            except RuntimeError:
                link = ""
            if CFG.get("share_to_story"):
                share_to_story(url, token)
            return media_id, link
        except (RuntimeError, subprocess.CalledProcessError) as e:
            last = f"{host.__name__}: {str(e)[:300]}"
            log(last)
        finally:
            unhost(path)
            if hosted_cover:
                unhost_release(cover)
    raise RuntimeError(f"Instagram did not accept the video ({last})")


def post_ready_made(listing):
    """Post videos that were made by hand and uploaded to the hosting release beforehand."""
    token = refresh_token(os.environ["IG_TOKEN"])
    user = os.environ["IG_USER_ID"]
    state = load_state()
    done = {p["id"] for p in state["posts"]}
    failed = []
    for item in json.load(open(os.path.join(ROOT, listing))):
        if item["id"] in done:
            log(f"{item['id']} is already posted; skipping")
            continue
        url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{item['file']}"
        try:
            container = ig("POST", f"{user}/media", token, media_type="REELS", video_url=url,
                           caption=item["caption"])["id"]
            problem = wait_container(container, token)
            if problem:
                raise RuntimeError(f"Instagram said {problem}")
            media_id = ig("POST", f"{user}/media_publish", token, creation_id=container)["id"]
            try:
                link = ig("GET", media_id, token, fields="permalink").get("permalink", "")
            except RuntimeError:
                link = ""
            if CFG.get("share_to_story"):
                share_to_story(url, token)
            log(f"Published {item['id']}: {link or media_id} (id {media_id})")
            state["posts"].append({k: item[k] for k in ("id", "title", "hook", "fact", "narration", "caption")}
                                  | {"media_id": media_id, "permalink": link,
                                     "posted_at": now().isoformat(timespec="seconds")})
            save_state(state)
            gh("release", "delete-asset", RELEASE_TAG, item["file"], "--repo", REPO, "-y", check=False)
        except RuntimeError as e:
            log(f"{item['id']} FAILED: {str(e)[:300]}")
            failed.append(item["id"])
    if failed:
        raise RuntimeError(f"not posted: {', '.join(failed)}")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="post to Instagram (otherwise render only)")
    ap.add_argument("--reuse-render", action="store_true",
                    help="skip writing and rendering; use the Reel from the last run")
    ap.add_argument("--script-only", action="store_true", help="write the script and stop")
    ap.add_argument("--scheduled", action="store_true",
                    help="started by the timer: skip if this half of the day already has a Reel")
    ap.add_argument("--list-models", action="store_true", help="print the Gemini models this key can use")
    ap.add_argument("--format", default="auto", choices=["auto", *FORMATS],
                    help="which kind of Reel to make; auto = by time of day (morning regular, afternoon quiz, evening three)")
    ap.add_argument("--post-files", metavar="JSON",
                    help="post ready-made videos already uploaded to the hosting release (hand-made Reels)")
    args = ap.parse_args()

    if args.list_models:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60)
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/")[-1])
        return

    if args.post_files:
        return post_ready_made(args.post_files)

    state = load_state()
    if args.scheduled:
        # GitHub's timer is unreliable, so it is set to fire several times per slot.
        # The first run that succeeds posts; the later ones stop here.
        slot = ist_slot
        # GitHub's spare timer can fire many hours late; never post in the middle of the night.
        ist_hour = now().astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))).hour
        if not 10 <= ist_hour < 22:
            log("Outside posting hours (10:00-22:00 India time); nothing to do")
            return
        if any(slot(dt.datetime.fromisoformat(p["posted_at"])) == slot(now()) for p in state["posts"]):
            log("This slot already has a Reel; nothing to do")
            return
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
        job_id = now().strftime("%Y%m%d%H%M")
        fmt = FORMATS[ist_slot(now())[1]] if args.format == "auto" else args.format
        log(f"Format: {fmt}")
        if fmt == "quiz":
            plan = write_quiz(state, job_id)
        elif fmt == "three":
            plan = write_three(state)
        else:
            plan = write_script(state)
        if plan.get("segments"):
            plan["narration"] = " ".join(seg["say"] for seg in plan["segments"])
        tags = " ".join("#" + str(t).lstrip("#").replace(" ", "") for t in plan["hashtags"])
        pending = {
            "id": job_id,
            "title": plan["title"],
            "hook": plan["narration"].split(". ")[0][:120],
            "fact": plan["fact"],
            "narration": plan["narration"],
            # Models sometimes write the two characters "\\n" instead of a real line break.
            "caption": (f"{plan['caption'].replace(chr(92) + 'n', chr(10)).strip()}\n\n"
                        f"{CFG['caption_follow_line']}\n\n{tags} #aigenerated"),
            # At most 6 words on screen and 3 on the cover, whatever the writer returned.
            "hook_text": " ".join(plan["hook_text"].split()[:6]),
            "cover_text": " ".join(plan["cover_text"].split()[:3]),
        }
        log(f"Topic: {plan['title']}")
        log(f"Narration: {plan['narration']}")
        log(f"Hook on screen: {pending['hook_text']} | Cover: {pending['cover_text']}")
        log(f"Caption:\n{pending['caption']}")
        for i, s in enumerate(plan.get("scenes", [])[:CFG["scenes"]], 1):
            log(f"Scene {i}: {s['prompt']}")
        for i, seg in enumerate(plan.get("segments", []), 1):
            shown = seg.get("pics") or [f"{label}: {prompt}" for prompt, label in seg["split"]]
            log(f"Part {i} [{seg.get('header', '')}] {seg['say']} || " + " | ".join(shown))
        if args.script_only:
            return
        state["pending"] = pending
        save_state(state)
        job = {"id": job_id, "voice": CFG["voice"], "quality": CFG["quality"], "visual": CFG["visual"],
               "hook_text": pending["hook_text"], "cover_text": pending["cover_text"]}
        if plan.get("segments"):
            job.update(segments=plan["segments"], mood=plan["mood"])
        else:
            job.update(narration=plan["narration"],
                       scenes=[{"prompt": s["prompt"]} for s in plan["scenes"][:CFG["scenes"]]])
        render(job)
        video = fetch_output(job_id)
        if not video:
            raise RuntimeError("Kaggle finished but produced no Reel file")
    log(f"Reel ready: {os.path.basename(video)} ({os.path.getsize(video) / 1e6:.1f} MB)")

    if not args.publish:
        log("Render-only run: not posting. Download the video from this run's artifacts.")
        return

    cover = os.path.join(OUT_DIR, f"cover-{pending['id']}.jpg")
    media_id, link = publish(video, pending["caption"], token, cover if os.path.exists(cover) else None)
    log(f"Published: {link or media_id} (id {media_id})")
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
