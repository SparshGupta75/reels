# Reel maker: runs on a free Kaggle GPU (T4) and turns one job into a finished 9:16 Reel.
#
# Job = narration + scene prompts (written by Gemini in the n8n workflow).
# Steps: Kokoro voiceover -> faster-whisper word timings -> Wan 2.2 TI2V-5B clips
#        -> FFmpeg cut/upscale/captions/audio -> /kaggle/working/reel-<id>.mp4
#
# n8n replaces __JOB_B64__ below with the base64 job JSON before pushing this script.
# Run it as-is (placeholder untouched) to render a built-in demo job.

import base64
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time

JOB_B64 = "__JOB_B64__"

DEMO_JOB = {
    "id": "demo",
    "narration": (
        "That fluffy cloud above you weighs five hundred tonnes. "
        "That's about a hundred elephants, floating over your head. "
        "So why doesn't it fall? Its water is spread across billions of droplets, "
        "each so tiny that rising warm air is enough to hold it up. "
        "Follow for a fact like this every day."
    ),
    "scenes": [
        {"prompt": "Low angle looking straight up at one enormous white cumulus cloud with puffy rounded tops filling the frame against a deep blue sky, slow drifting motion, cinematic, photorealistic"},
        {"prompt": "A gigantic towering white cumulus cloud with a flat grey base looming over tiny green hills far below, the cloud fills most of the frame, slow push-in, golden sunlight, cinematic, photorealistic"},
        {"prompt": "Extreme macro shot inside a white cloud, countless tiny glistening water droplets floating in mist, backlit by soft sunlight, slow motion, shallow depth of field, photorealistic"},
        {"prompt": "Time-lapse of a large white cumulus cloud billowing and rising upward on warm air above sunlit fields, the cloud centred and filling the frame, blue sky, cinematic, photorealistic"},
    ],
    "voice": "am_michael",
    "quality": "fast",
}

# --- settings ---------------------------------------------------------------
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
GEN_FPS = 24                       # Wan 2.2 5B is trained at 24 fps
OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30
QUALITY = {                        # (width, height, steps) of the generated clips
    "fast": (480, 832, 30),        # ~a few minutes per clip on a T4 (estimate)
    "high": (704, 1280, 40),       # sharper, roughly 3x slower
}
NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, text, subtitles, logo, "
    "static frame, jpeg artifacts, extra limbs, bad anatomy, overexposed"
)
IMAGE_MODEL = "SG161222/RealVisXL_V5.0"     # photoreal SDXL; chosen in a side-by-side test on a T4
IMAGE_NEGATIVE = ("blurry, low quality, deformed, distorted, watermark, text, letters, logo, cartoon, "
                  "illustration, painting, extra limbs, bad anatomy, duplicate, frame, border")
MAX_CLIP_FRAMES = 121              # ~5 s; longer shots get slowed down slightly

WORK = "/tmp/reel"
FONT_DIR = WORK + "/fonts"
TITLE_FONT, TITLE_BOLD, HOOK_SIZE, COVER_SIZE = "DejaVu Sans", -1, 118, 180
COVER_TAG = "MIND BLOWN REELS"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("out")
FAKE_VIDEO = os.environ.get("REEL_FAKE_VIDEO") == "1"   # local test: colour bars instead of AI clips
KOKORO_ENV = "/tmp/kokoro-env"
KOKORO_PY = KOKORO_ENV + "/bin/python"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_job():
    if JOB_B64.startswith("__"):
        log("No job injected, rendering the demo job")
        job = dict(DEMO_JOB)
    else:
        job = json.loads(base64.b64decode(JOB_B64).decode("utf-8"))
    job.setdefault("id", time.strftime("%Y%m%d-%H%M%S"))
    job.setdefault("voice", "am_michael")
    job.setdefault("quality", "fast")
    if not job.get("narration") or not job.get("scenes"):
        raise ValueError("job needs 'narration' and a non-empty 'scenes' list")
    return job


def install_deps():
    pkgs = ["faster-whisper", "ftfy"]
    if not FAKE_VIDEO:
        # Exact versions from a run that worked; newer releases have broken on Kaggle before.
        pkgs += ["diffusers==0.41.0", "transformers==5.18.0", "accelerate==1.15.0"]
    run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
    if not FAKE_VIDEO:
        # Kaggle's torchao is too old for new diffusers and breaks its import; it is unused here.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        log("Versions: " + ", ".join(x for x in frozen.split() if x.split("==")[0] in
                                     ("diffusers", "transformers", "accelerate", "torch")))
    # Kokoro needs Python < 3.13 and Kaggle runs 3.13, so it gets its own 3.12 env.
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    if not os.path.exists(KOKORO_PY):
        run([sys.executable, "-m", "uv", "venv", "-q", "--seed", "--python", "3.12", KOKORO_ENV])
    run([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", KOKORO_PY,
         "--torch-backend", "cpu", "kokoro>=0.9.4", "soundfile",
         "transformers>=4.40,<5"])   # without the pin uv picks a 2021 transformers
    # espeak-ng improves pronunciation of unusual words; Kokoro works without it.
    if shutil.which("espeak-ng") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y espeak-ng",
                       shell=True, check=False)
    if shutil.which("ffmpeg") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg",
                       shell=True, check=True)


# --- 1. voiceover -----------------------------------------------------------
KOKORO_SCRIPT = r"""
import sys
import numpy as np
import soundfile as sf
from kokoro import KPipeline

text, voice, path = sys.argv[1:4]
lang = voice[0] if voice[0] in "ab" else "a"   # a = US English, b = UK English
pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
parts = []
for result in pipeline(text, voice=voice, speed=1.08):
    audio = result.audio if hasattr(result, "audio") else result[2]
    if audio is not None:
        parts.append(audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio))
sf.write(path, np.concatenate(parts), 24000)
"""


def make_voice(text, voice, path):
    subprocess.run([KOKORO_PY, "-c", KOKORO_SCRIPT, text, voice, path], check=True)
    return duration(path)


# --- 2. word timings for captions ---------------------------------------------
def word_timings(wav_path, narration):
    """Time every word of the narration. Whisper listens to the voiceover for the timing, but
    the words shown are always the script's own: Whisper sometimes skips whole sentences or
    mishears words, so its output is only lined up against the script, never shown directly."""
    import difflib

    import numpy as np
    from faster_whisper import WhisperModel

    # Decode with ffmpeg: faster-whisper's own decoder breaks on Kaggle's PyAV version.
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", wav_path, "-f", "f32le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    total = len(audio) / 16000
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=False,
                                   condition_on_previous_text=False)
    heard = [(w.start, w.end, w.word) for s in segments for w in s.words if w.word.strip()]
    del model
    gc.collect()

    def norm(w):
        return "".join(c for c in w.lower() if c.isalnum())

    script = narration.split()
    times = [None] * len(script)
    matcher = difflib.SequenceMatcher(a=[norm(w) for w in script], b=[norm(h[2]) for h in heard],
                                      autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            times[block.a + k] = heard[block.b + k][:2]
    log(f"Caption timing: {sum(t is not None for t in times)} of {len(script)} words heard clearly")

    # Words Whisper missed share the gap between their timed neighbours, by length.
    i = 0
    while i < len(script):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(script) and times[j] is None:
            j += 1
        start = times[i - 1][1] if i else 0.0
        end = times[j][0] if j < len(script) else total
        end = max(end, start + 0.05 * (j - i))
        weights = [len(w) + 1 for w in script[i:j]]
        t = start
        for k in range(i, j):
            step = (end - start) * weights[k - i] / sum(weights)
            times[k] = (t, t + step)
            t += step
        i = j
    return [(times[k][0], times[k][1], script[k]) for k in range(len(script))]


def ass_time(t):
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


YELLOW = "&H0AD6FF&"     # ASS colours are BGR


def mark_up(text):
    """Turn "SQUIRRELS HAVE *BACKWARD* ANKLES" into ASS text with the starred word in yellow."""
    text = text.upper().replace("{", "(").replace("}", ")").replace("\n", "\\N")
    parts = text.split("*")
    if len(parts) != 3:
        return text.replace("*", "")
    return parts[0] + "{\\c" + YELLOW + "}" + parts[1] + "{\\c&HFFFFFF&}" + parts[2]


def make_ass(words, path, hook="", max_words=3, max_chars=16):
    """Bold, centred, 2-3 word captions with a small pop-in, TikTok style, plus an
    optional big hook line across the top for the first two seconds."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,96,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,560,1
Style: Hook,{TITLE_FONT},{HOOK_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,{TITLE_BOLD},0,0,0,100,100,1,0,1,10,4,8,70,70,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    groups, cur = [], []
    for w in words:
        text = " ".join(x[2] for x in cur + [w])
        if cur and (len(cur) >= max_words or len(text) > max_chars
                    or cur[-1][2].endswith((".", ",", "?", "!", ":"))):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    lines = []
    if hook.strip():
        hw = hook.split()
        if len(hw) > 2:                       # two balanced lines read better than one long one
            hook = " ".join(hw[:(len(hw) + 1) // 2]) + "\n" + " ".join(hw[(len(hw) + 1) // 2:])
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(2.2)},Hook,,0,0,0,,"
                     + r"{\fad(0,250)}" + mark_up(hook))
    for i, g in enumerate(groups):
        start = g[0][0]
        end = groups[i + 1][0][0] if i + 1 < len(groups) else g[-1][1] + 0.3
        end = min(end, g[-1][1] + 0.6)
        text = " ".join(x[2] for x in g).upper().replace("{", "(").replace("}", ")")
        pop = r"{\fscx85\fscy85\t(0,90,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{pop}{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def make_cover(joined, at, text, path):
    """A still for the profile grid: a clean frame (no captions) with two or three huge words."""
    stacked = mark_up(text).replace(" ", "\\N")     # one word per line
    ass = os.path.join(WORK, "cover.ass")
    with open(ass, "w", encoding="utf-8") as f:
        f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Big,{TITLE_FONT},{COVER_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,{TITLE_BOLD},0,0,0,100,100,1,0,1,13,5,5,60,60,0,1
Style: Tag,DejaVu Sans,44,&H00000000,&H000000FF,{YELLOW[:-1].replace("&H", "&H00")},{YELLOW[:-1].replace("&H", "&H00")},-1,0,0,0,100,100,2,0,3,14,0,8,60,60,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,9:00:00.00,Tag,,0,0,0,,{COVER_TAG}
Dialogue: 0,0:00:00.00,9:00:00.00,Big,,0,0,0,,{{\\pos({OUT_W // 2},{int(OUT_H * 0.66)})}}{stacked}
""")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", joined,
         "-vf", f"eq=brightness=-0.06:saturation=1.1,{ass_filter(ass)}", "-frames:v", "1", "-q:v", "2", path])


def ass_filter(ass_path):
    return f"ass={ass_path}" + (f":fontsdir={FONT_DIR}" if os.listdir(FONT_DIR) else "")


def setup_fonts():
    """Anton is the tall poster typeface used for hooks and covers; fall back to DejaVu Bold."""
    global TITLE_FONT, TITLE_BOLD, HOOK_SIZE, COVER_SIZE
    os.makedirs(FONT_DIR, exist_ok=True)
    for src in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(src):
            shutil.copy(src, FONT_DIR)
    anton = os.path.join(FONT_DIR, "Anton-Regular.ttf")
    try:
        import urllib.request
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf", anton)
        if os.path.getsize(anton) < 50_000:
            raise OSError("font download too small")
        TITLE_FONT, TITLE_BOLD, HOOK_SIZE, COVER_SIZE = "Anton", 0, 164, 260
    except Exception as e:
        log(f"Poster font unavailable ({e}); using DejaVu Bold")
        if os.path.exists(anton):
            os.remove(anton)


# --- 3. AI video clips --------------------------------------------------------
def frames_for(seconds):
    n = min(math.ceil(seconds * GEN_FPS), MAX_CLIP_FRAMES)
    return max(4 * math.ceil((n - 1) / 4) + 1, 17)   # Wan needs 4k+1 frames


def make_clips(prompts, seconds, quality):
    if FAKE_VIDEO:
        return fake_clips(prompts, seconds)

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    from transformers import UMT5EncoderModel

    from concurrent.futures import ThreadPoolExecutor

    width, height, steps = QUALITY.get(quality, QUALITY["fast"])
    devices = [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 2))]
    log(f"GPUs: {len(devices)} x {torch.cuda.get_device_name(0)}")
    count = len(prompts)

    def run_on_all(work):
        """Run work(w) once per GPU at the same time; GPU w handles clips w, w+N, ..."""
        with ThreadPoolExecutor(len(devices)) as ex:
            for future in [ex.submit(work, w) for w in range(len(devices))]:
                future.result()

    def load_pipe(dev, **kw):
        class Pipe(WanPipeline):
            # The parts live on different devices, so say where denoising happens.
            @property
            def _execution_device(self):
                return torch.device(dev)

        pipe = Pipe.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16, **kw)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    # T4 has no fast bf16: transformer runs in fp16. The T5 text encoder overflows in
    # fp16, so it runs once in bf16 (slow on a T4, far slower on the CPU) and is then
    # thrown away to make room for the transformer.
    te = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipes = [load_pipe(devices[0], text_encoder=te)]

    log("Encoding prompts")
    te.to(devices[0])
    embeds = []
    with torch.no_grad():
        for prompt in prompts:
            pe, ne = pipes[0].encode_prompt(prompt=prompt, negative_prompt=NEGATIVE,
                                            do_classifier_free_guidance=True,
                                            max_sequence_length=512,
                                            device=torch.device(devices[0]), dtype=torch.float16)
            embeds.append((pe.cpu(), ne.cpu()))
    pipes[0].text_encoder = None
    del te
    gc.collect()
    torch.cuda.empty_cache()

    # Step 1: one transformer per GPU turns the prompts into latents, in parallel.
    pipes[0].transformer.to(devices[0])
    for dev in devices[1:]:
        pipes.append(load_pipe(dev, text_encoder=None))
        pipes[-1].transformer.to(dev)
    z = vae.config.z_dim
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1)
    lat_std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1)
    latents = [None] * count

    def denoise(w):
        dev = devices[w]
        for i in range(w, count, len(devices)):
            n = frames_for(seconds[i])
            log(f"Clip {i + 1}/{count} on {dev}: {width}x{height}, {n} frames, {steps} steps")
            t0 = time.time()
            pe, ne = embeds[i]
            out = pipes[w](prompt_embeds=pe.to(dev), negative_prompt_embeds=ne.to(dev),
                           height=height, width=width, num_frames=n,
                           guidance_scale=5.0, num_inference_steps=steps, output_type="latent",
                           generator=torch.Generator(dev).manual_seed(1000 + i)).frames
            if not torch.isfinite(out).all():
                raise RuntimeError(f"clip {i + 1}: NaN/inf latents (fp16 overflow)")
            latents[i] = out.float().cpu() * lat_std + lat_mean
            log(f"Clip {i + 1} denoised in {time.time() - t0:.0f}s")

    # PyTorch sets up its linear-algebra library the first time it is used, and crashes
    # ("lazy wrapper should be called at most once") if two threads trigger that at the
    # same moment. Trigger it here, once, before the threads start.
    for dev in ["cpu", *devices]:
        torch.linalg.solve(torch.eye(2, device=dev), torch.ones(2, 1, device=dev))
        torch.linalg.inv(torch.eye(2, device=dev))

    run_on_all(denoise)

    # Step 2: a transformer and a VAE decode don't fit on one T4 together, so drop the
    # transformers first. Decoding on two GPUs from threads crashed CUDA, so each GPU
    # gets its own process.
    for i, lat in enumerate(latents):
        torch.save(lat, os.path.join(WORK, f"latent{i}.pt"))
    pipes.clear()
    del vae, latents
    gc.collect()
    torch.cuda.empty_cache()

    run_per_gpu("--decode", count, len(devices))
    return [os.path.join(WORK, f"clip{i}.mp4") for i in range(count)]


def run_per_gpu(flag, count, gpus):
    """Run this script once per GPU as its own process (threads sharing CUDA crashed);
    GPU w handles items w, w+gpus, ... A failed worker gets one more try."""
    def start(w):
        mine = [str(i) for i in range(w, count, gpus)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(w)}
        script = os.path.abspath(globals().get("__file__") or sys.argv[0])
        return subprocess.Popen([sys.executable, script, flag, *mine], env=env)

    workers = [start(w) for w in range(min(gpus, count))]
    for w, proc in enumerate(workers):
        if proc.wait() != 0:
            log(f"Worker on GPU {w} failed, trying once more")
            if start(w).wait() != 0:
                raise RuntimeError(f"{flag} failed on GPU {w}")


def decode_worker(indices):
    """Runs in its own process on one GPU: turn saved latents into clip files."""
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    vae.enable_tiling()
    processor = VideoProcessor(vae_scale_factor=16)
    for i in indices:
        path = os.path.join(WORK, f"clip{i}.mp4")
        if os.path.exists(path):
            continue
        t0 = time.time()
        latents = torch.load(os.path.join(WORK, f"latent{i}.pt")).to("cuda")
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0].cpu()
        frames = processor.postprocess_video(video, output_type="np")[0]
        export_to_video(frames, path + ".tmp.mp4", fps=GEN_FPS)
        os.replace(path + ".tmp.mp4", path)
        log(f"Clip {i + 1} decoded in {time.time() - t0:.0f}s")
        del video, latents
        torch.cuda.empty_cache()


# --- 3b. still images with slow camera moves --------------------------------------
MOVES = [   # (zoom, x, y) expressions for ffmpeg's zoompan; N is the number of frames
    ("1+0.14*on/N", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),              # push in
    ("1.14", "iw/2-(iw/zoom/2)", "(ih-ih/zoom)*(1-on/N)"),                # drift up
    ("1.14-0.14*on/N", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),           # pull out
    ("1.14", "(iw-iw/zoom)*on/N", "ih/2-(ih/zoom/2)"),                    # drift right
    ("1+0.14*on/N", "iw/2-(iw/zoom/2)", "(ih-ih/zoom)*0.35"),             # push in, a little high
    ("1.14", "(iw-iw/zoom)*(1-on/N)", "ih/2-(ih/zoom/2)"),                # drift left
]


def still_clips(images, seconds):
    """Turn each still image into a clip with a slow zoom or drift."""
    paths = []
    for i, (img, secs) in enumerate(zip(images, seconds)):
        n = math.ceil(secs * OUT_FPS) + 2
        z, x, y = (e.replace("N", str(n)) for e in MOVES[i % len(MOVES)])
        path = os.path.join(WORK, f"clip{i}.mp4")
        # Zooming a 2x enlarged copy keeps the motion smooth instead of jittery.
        vf = (f"scale={OUT_W * 2}:{OUT_H * 2}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={OUT_W * 2}:{OUT_H * 2},"
              f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={OUT_W}x{OUT_H}:fps={OUT_FPS},format=yuv420p")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", img, "-vf", vf, "-frames:v", str(n),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", path])
        paths.append(path)
    return paths


def fake_stills(prompts):
    paths = []
    for i, _ in enumerate(prompts):
        path = os.path.join(WORK, f"img{i}.jpg")
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1088x1920:rate=1",
             "-frames:v", "1", path])
        paths.append(path)
    return paths


def make_stills(prompts):
    if FAKE_VIDEO:
        return fake_stills(prompts)
    import torch

    with open(os.path.join(WORK, "prompts.json"), "w") as f:
        json.dump(prompts, f)
    gpus = max(torch.cuda.device_count(), 1)
    log(f"Making {len(prompts)} pictures with {IMAGE_MODEL} on {gpus} GPU(s)")
    run_per_gpu("--stills", len(prompts), gpus)
    return [os.path.join(WORK, f"img{i}.jpg") for i in range(len(prompts))]


def stills_worker(indices):
    """Runs in its own process on one GPU: draw the pictures for the given scenes."""
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    prompts = json.load(open(os.path.join(WORK, "prompts.json")))
    # SDXL's own VAE overflows in fp16; this fixed one does not.
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(IMAGE_MODEL, vae=vae, torch_dtype=torch.float16,
                                                         variant="fp16")
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(IMAGE_MODEL, vae=vae, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    refine = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    refine.set_progress_bar_config(disable=True)
    for i in indices:
        path = os.path.join(WORK, f"img{i}.jpg")
        if os.path.exists(path):
            continue
        t0 = time.time()
        g = torch.Generator("cuda").manual_seed(1000 + i)
        # Draw at the size the model was trained on, then redraw lightly at full Reel size for sharpness.
        img = pipe(prompt=prompts[i], negative_prompt=IMAGE_NEGATIVE, width=768, height=1344,
                   num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]
        img = refine(prompt=prompts[i], negative_prompt=IMAGE_NEGATIVE, image=img.resize((1088, 1920)),
                     strength=0.3, num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]
        img.save(path + ".tmp.jpg", quality=95)
        os.replace(path + ".tmp.jpg", path)
        log(f"Picture {i + 1} made in {time.time() - t0:.0f}s")


def fake_clips(prompts, seconds):
    paths = []
    for i, secs in enumerate(seconds):
        path = os.path.join(WORK, f"clip{i}.mp4")
        n = frames_for(secs)
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"testsrc2=size=480x832:rate={GEN_FPS}", "-frames:v", str(n),
             "-pix_fmt", "yuv420p", path])
        paths.append(path)
    return paths


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# --- 4. assembly ----------------------------------------------------------------
def assemble(clips, seconds, voice_wav, ass_path, out_path):
    segs = []
    for i, (clip, secs) in enumerate(zip(clips, seconds)):
        stretch = max(1.0, secs / duration(clip))    # slow down if the clip is too short
        seg = os.path.join(WORK, f"seg{i}.mp4")
        vf = (f"setpts={stretch:.4f}*PTS,"
              f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={OUT_W}:{OUT_H},fps={OUT_FPS},format=yuv420p")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vf", vf, "-t", f"{secs:.3f}",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", seg])
        segs.append(seg)

    concat = os.path.join(WORK, "concat.txt")
    with open(concat, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    joined = os.path.join(WORK, "joined.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat,
         "-c", "copy", joined])

    run(["ffmpeg", "-y", "-loglevel", "error", "-i", joined, "-i", voice_wav,
         "-vf", ass_filter(ass_path), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(OUT_FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-shortest", "-movflags", "+faststart", out_path])
    return joined


def main():
    if sys.argv[1:2] == ["--decode"]:
        return decode_worker([int(x) for x in sys.argv[2:]])
    if sys.argv[1:2] == ["--stills"]:
        return stills_worker([int(x) for x in sys.argv[2:]])
    t0 = time.time()
    job = load_job()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    install_deps()

    voice_wav = os.path.join(WORK, "voice.wav")
    speech = make_voice(job["narration"], job["voice"], voice_wav)
    total = speech + 0.3      # short tail so the end runs straight back into the start
    log(f"Voiceover: {speech:.1f}s")

    words = word_timings(voice_wav, job["narration"])
    ass_path = os.path.join(WORK, "captions.ass")
    setup_fonts()
    make_ass(words, ass_path, job.get("hook_text", ""))
    log(f"Captions: {len(words)} words")

    prompts = [s["prompt"] if isinstance(s, dict) else str(s) for s in job["scenes"]]
    seconds = [total / len(prompts)] * len(prompts)
    if job.get("visual") == "stills":
        clips = still_clips(make_stills(prompts), seconds)
    else:
        clips = make_clips(prompts, seconds, job["quality"])

    # Unique name so the pipeline never picks up an older run's video by mistake.
    out_path = os.path.join(OUT_DIR, f"reel-{job['id']}.mp4")
    joined = assemble(clips, seconds, voice_wav, ass_path, out_path)
    if job.get("cover_text"):
        # Second scene, a little way in: usually the clearest view of the subject.
        at = seconds[0] + seconds[1] * 0.4 if len(seconds) > 1 else seconds[0] * 0.5
        try:
            make_cover(joined, at, job["cover_text"], os.path.join(OUT_DIR, f"cover-{job['id']}.jpg"))
        except subprocess.CalledProcessError as e:
            log(f"Cover image failed ({e}); the Reel is fine without it")

    size_mb = os.path.getsize(out_path) / 1e6
    info = {"id": job["id"], "seconds": round(duration(out_path), 2), "size_mb": round(size_mb, 1),
            "render_minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(info, f)
    log(f"Done: {info}")
    if size_mb > 95:
        raise RuntimeError("the Reel is over Instagram's 100 MB limit")


if __name__ == "__main__":
    main()
