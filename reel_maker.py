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
MAX_CLIP_FRAMES = 121              # ~5 s; longer shots get slowed down slightly

WORK = "/tmp/reel"
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
        pkgs += ["diffusers>=0.35.0", "transformers", "accelerate"]
    run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
    if not FAKE_VIDEO:
        # Kaggle's torchao is too old for new diffusers and breaks its import; it is unused here.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        log("Versions: " + ", ".join(x for x in frozen.split() if x.split("==")[0] in
                                     ("diffusers", "transformers", "accelerate", "torch")))
    # Kokoro needs Python < 3.13 and Kaggle runs 3.13, so it gets its own 3.12 env.
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
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
    from faster_whisper import WhisperModel

    import numpy as np

    # Decode with ffmpeg: faster-whisper's own decoder breaks on Kaggle's PyAV version.
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", wav_path, "-f", "f32le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, word_timestamps=True,
                                   initial_prompt=narration, vad_filter=False)
    words = [(w.start, w.end, w.word.strip()) for s in segments for w in s.words if w.word.strip()]
    del model
    gc.collect()
    return words


def ass_time(t):
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def make_ass(words, path, max_words=3, max_chars=16):
    """Bold, centred, 2-3 word captions with a small pop-in, TikTok style."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,96,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,560,1

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
    for i, g in enumerate(groups):
        start = g[0][0]
        end = groups[i + 1][0][0] if i + 1 < len(groups) else g[-1][1] + 0.3
        end = min(end, g[-1][1] + 0.6)
        text = " ".join(x[2] for x in g).upper().replace("{", "(").replace("}", ")")
        pop = r"{\fscx85\fscy85\t(0,90,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{pop}{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


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

    def start(w):
        mine = [str(i) for i in range(w, count, len(devices))]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(w)}
        return subprocess.Popen([sys.executable, os.path.abspath(globals().get("__file__") or sys.argv[0]), "--decode", *mine], env=env)

    workers = [start(w) for w in range(len(devices))]
    for w, proc in enumerate(workers):
        if proc.wait() != 0:
            log(f"Decoder on GPU {w} failed, trying once more")
            if start(w).wait() != 0:
                raise RuntimeError(f"decoding failed on GPU {w}")
    return [os.path.join(WORK, f"clip{i}.mp4") for i in range(count)]


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

    fonts = "/usr/share/fonts/truetype/dejavu"
    ass_filter = f"ass={ass_path}" + (f":fontsdir={fonts}" if os.path.isdir(fonts) else "")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", joined, "-i", voice_wav,
         "-vf", ass_filter, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(OUT_FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-shortest", "-movflags", "+faststart", out_path])


def main():
    if sys.argv[1:2] == ["--decode"]:
        return decode_worker([int(x) for x in sys.argv[2:]])
    t0 = time.time()
    job = load_job()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    install_deps()

    voice_wav = os.path.join(WORK, "voice.wav")
    speech = make_voice(job["narration"], job["voice"], voice_wav)
    total = speech + 0.6
    log(f"Voiceover: {speech:.1f}s")

    words = word_timings(voice_wav, job["narration"])
    ass_path = os.path.join(WORK, "captions.ass")
    make_ass(words, ass_path)
    log(f"Captions: {len(words)} words")

    prompts = [s["prompt"] if isinstance(s, dict) else str(s) for s in job["scenes"]]
    seconds = [total / len(prompts)] * len(prompts)
    clips = make_clips(prompts, seconds, job["quality"])

    # Unique name so n8n never picks up an older run's video by mistake.
    out_path = os.path.join(OUT_DIR, f"reel-{job['id']}.mp4")
    assemble(clips, seconds, voice_wav, ass_path, out_path)

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
