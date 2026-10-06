# reels

Makes one short narrated fact video a day and posts it to Instagram as a Reel, using only free services.

- `.github/workflows/daily-reel.yml` runs every day on GitHub Actions.
- `pipeline/post_reel.py` writes the script (Gemini), renders it on a Kaggle GPU with `reel_maker.py`, posts it to Instagram, and records how past Reels performed so later scripts can learn from them.
- `pipeline/config.json` holds the settings (topic area, number of scenes, voice).
- `pipeline/system_prompt.txt` holds the writing rules.
- `state/history.json` is the log of posted Reels and their results.

Keys live in the repository's Actions secrets: `GEMINI_API_KEY`, `KAGGLE_API_TOKEN`, `IG_USER_ID`, `IG_TOKEN`, and optionally `SECRETS_PAT`.
The daily schedule only posts when the repository variable `AUTO_POST` is `true`.
