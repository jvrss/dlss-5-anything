#!/usr/bin/env python3
"""
process_video.py — Apply FLUX.2 Klein 9B KV to every frame of a video.

Usage:
    python process_video.py [--video VIDEO_PATH] [--output OUTPUT_PATH]
                            [--steps N] [--seed N] [--prompt TEXT]
                            [--no-audio] [--resume]

Features
--------
• Streams raw frames from ffmpeg — no massive raw-frame dump to disk
• Saves enhanced frames as JPEG (quality 95) for resume support
• Resume: already-processed frames are detected and skipped automatically
• Per-frame seed (SEED + frame_index) for full reproducibility across runs
• Rich progress bars, live stats panel, per-milestone console logs
• Graceful Ctrl-C: saves progress, shows summary, then exits cleanly
• Reassembles final video (libx264) with original audio track
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Enhance every video frame with FLUX.2 Klein KV")
parser.add_argument("--video",   default="video/ignition.mkv", help="Input video path")
parser.add_argument("--output",  default="output_enhanced.mp4", help="Output video path")
parser.add_argument("--frames",  default="frames_enhanced", help="Dir for enhanced JPEG frames")
parser.add_argument("--steps",   type=int, default=4, help="Diffusion inference steps (default 4)")
parser.add_argument("--seed",    type=int, default=42, help="Base seed (frame_seed = seed + frame_idx)")
parser.add_argument("--prompt",  default="make it more realistic", help="Prompt for FLUX model")
parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved frames (1-95)")
parser.add_argument("--no-audio", dest="no_audio", action="store_true", help="Discard audio track")
parser.add_argument("--resume",  action="store_true", default=True, help="Skip existing frames (default: on)")
args = parser.parse_args()

VIDEO_IN     = Path(args.video)
OUTPUT_VIDEO = Path(args.output)
ENHANCED_DIR = Path(args.frames)
STEPS        = args.steps
BASE_SEED    = args.seed
PROMPT       = args.prompt
JPEG_QUALITY = args.quality
KEEP_AUDIO   = not args.no_audio

HF_TOKEN = os.environ.get("HF_TOKEN")

# ─── patch diffusers before import ───────────────────────────────────────────
def apply_patch():
    try:
        import diffusers
    except ImportError:
        return
    site_packages = os.path.dirname(diffusers.__file__)
    patch_file = Path(__file__).parent / "flux2_klein_kv.patch"
    if not patch_file.exists():
        return
    for cmd in (
        ["patch", "-p2", "--forward", "--batch"],
        ["git", "apply", "-p2", "--ignore-whitespace",
         "--whitespace=nowarn", str(patch_file)],
    ):
        try:
            if cmd[0] == "patch":
                r = subprocess.run(cmd, cwd=os.path.dirname(site_packages),
                                   stdin=open(patch_file), capture_output=True, text=True)
            else:
                r = subprocess.run(cmd, cwd=os.path.dirname(site_packages),
                                   capture_output=True, text=True)
            if r.returncode == 0:
                print("✓ Patch applied successfully")
                return
        except FileNotFoundError:
            continue
    print("⚠  Could not apply patch — continuing anyway")

apply_patch()

# ─── heavy imports (after patch) ─────────────────────────────────────────────
import numpy as np
import torch
from PIL import Image

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-kv"
dtype    = torch.bfloat16
device   = "cuda" if torch.cuda.is_available() else "cpu"

console = Console()

# ─── helpers ──────────────────────────────────────────────────────────────────

def get_video_info(path: Path):
    """Return (fps, width, height, has_audio) via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    data = json.loads(r.stdout)
    vs       = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    fps = num / den
    return fps, int(vs["width"]), int(vs["height"]), has_audio


def count_total_frames(path: Path) -> int:
    """Count exact frame count (slow — scans whole file)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    txt = r.stdout.strip()
    return int(txt) if txt.isdigit() else 0


def target_resolution(iw: int, ih: int):
    """Scale to max 1024 px, multiple of 16, preserve aspect ratio."""
    ar = iw / ih
    if ar >= 1:
        ow = 1024
        oh = round(1024 / ar / 16) * 16
    else:
        oh = 1024
        ow = round(1024 * ar / 16) * 16
    return max(256, min(1024, ow)), max(256, min(1024, oh))


def enhance_frame(pipe, img: Image.Image, seed: int) -> Image.Image:
    """Run the FLUX pipeline on one PIL image; returns enhanced PIL image."""
    ow, oh = target_resolution(*img.size)
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=PROMPT,
        image=[img],
        height=oh,
        width=ow,
        num_inference_steps=STEPS,
        generator=gen,
    ).images[0]


def reassemble_video(fps: float, has_audio: bool):
    """Encode enhanced JPEG frames back into mp4 with optional audio."""
    console.print("\n[bold cyan]═══ Reassembling video ═══[/bold cyan]")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(ENHANCED_DIR / "frame_%06d.jpg"),
    ]
    if has_audio and KEEP_AUDIO:
        cmd += ["-i", str(VIDEO_IN), "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(OUTPUT_VIDEO),
    ]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        console.print(f"[red bold]ffmpeg error:[/red bold]\n{r.stderr}")
        sys.exit(1)
    size_mb = OUTPUT_VIDEO.stat().st_size / 1_048_576
    console.print(
        f"[bold green]✓ Video saved → {OUTPUT_VIDEO}  ({size_mb:.1f} MB)[/bold green]"
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    wall_start = time.time()

    if not VIDEO_IN.exists():
        console.print(f"[red]Input video not found: {VIDEO_IN}[/red]")
        sys.exit(1)

    ENHANCED_DIR.mkdir(exist_ok=True)

    # ── header ──────────────────────────────────────────────────────────────
    console.rule("[bold bright_green]DLSS 5 Video Processor[/bold bright_green]")

    # ── video metadata ───────────────────────────────────────────────────────
    console.print("[cyan]Reading video metadata…[/cyan]")
    fps, vw, vh, has_audio = get_video_info(VIDEO_IN)

    console.print(f"[cyan]Counting total frames (scanning file)…[/cyan]  ", end="")
    total_frames = count_total_frames(VIDEO_IN)
    if total_frames == 0:
        total_frames = 20496  # fallback known value
        console.print(f"[yellow]could not count — using estimate {total_frames}[/yellow]")
    else:
        console.print(f"[green]{total_frames}[/green]")

    out_w, out_h = target_resolution(vw, vh)

    # ── info table ───────────────────────────────────────────────────────────
    info = Table(box=box.ROUNDED, border_style="green", show_header=False, padding=(0, 1))
    info.add_column("", style="cyan bold", no_wrap=True)
    info.add_column("", style="white")
    info.add_row("Input",         str(VIDEO_IN))
    info.add_row("Source res",    f"{vw}x{vh}")
    info.add_row("Source FPS",    f"{fps:.2f}")
    info.add_row("Total frames",  str(total_frames))
    info.add_row("Audio",         "✓ yes" if has_audio else "✗ no")
    info.add_row("Output res",    f"{out_w}x{out_h}  (model max 1024 px)")
    info.add_row("Model",         MODEL_ID)
    info.add_row("Prompt",        PROMPT)
    info.add_row("Steps",         str(STEPS))
    info.add_row("Base seed",     str(BASE_SEED))
    info.add_row("Device",        device)
    info.add_row("Output",        str(OUTPUT_VIDEO))
    info.add_row("Frames dir",    str(ENHANCED_DIR))
    console.print(info)

    # ── check existing / resume ──────────────────────────────────────────────
    existing = {f.stem for f in ENHANCED_DIR.glob("frame_*.jpg")}
    skipped_count = len(existing)
    if skipped_count:
        console.print(
            f"\n[yellow]Resume: {skipped_count} frames already processed — will skip.[/yellow]"
        )

    # ── load model ───────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Loading FLUX.2 Klein 9B KV…[/bold cyan]")
    t_load = time.time()
    pipe = Flux2KleinKVPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=HF_TOKEN)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    console.print(f"[green]✓ Model loaded in {time.time()-t_load:.1f}s[/green]")

    # ── open ffmpeg frame stream ─────────────────────────────────────────────
    frame_bytes = vw * vh * 3  # RGB24

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-i", str(VIDEO_IN),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # ── processing loop ──────────────────────────────────────────────────────
    frame_times: list[float] = []
    errors:      list[tuple] = []
    processed_count = 0
    frame_idx = 0
    interrupted = False

    LOG_INTERVAL = 50  # console log every N frames

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=38),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        TextColumn("[cyan]{task.fields[speed]}[/cyan]"),
        console=console,
        transient=False,
    )

    task = progress.add_task(
        "[green]Enhancing frames",
        total=total_frames,
        speed="—",
    )
    progress.update(task, advance=skipped_count)

    console.rule("[dim]Processing start[/dim]")
    progress.start()

    try:
        while True:
            raw = ffmpeg_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break  # end of stream

            frame_idx += 1
            stem     = f"frame_{frame_idx:06d}"
            out_path = ENHANCED_DIR / f"{stem}.jpg"

            # ── skip if already processed ──────────────────────────────────
            if stem in existing:
                continue  # progress already advanced above

            # ── process ───────────────────────────────────────────────────
            t0 = time.time()
            try:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(vh, vw, 3)
                img = Image.fromarray(arr)
                result = enhance_frame(pipe, img, seed=BASE_SEED + frame_idx)
                result.save(out_path, format="JPEG", quality=JPEG_QUALITY)
                processed_count += 1
            except Exception as exc:
                errors.append((frame_idx, str(exc)))
                console.log(f"[red]✗ Frame {frame_idx} error: {exc}[/red]")

            elapsed = time.time() - t0
            frame_times.append(elapsed)

            recent = frame_times[-20:]
            avg = sum(recent) / len(recent)
            speed_str = f"{60/avg:.1f} fr/min"
            progress.update(task, advance=1, speed=speed_str)

            # ── milestone log ──────────────────────────────────────────────
            if frame_idx % LOG_INTERVAL == 0:
                done_total  = skipped_count + processed_count
                remaining   = total_frames - done_total
                eta_min     = avg * remaining / 60
                wall_elapsed = (time.time() - wall_start) / 60
                console.log(
                    f"[dim]Frame [white]{frame_idx:>6}[/white]/{total_frames}"
                    f"  |  [white]{elapsed:.2f}s[/white] this frame"
                    f"  |  avg [white]{avg:.2f}s[/white]"
                    f"  |  [white]{speed_str}[/white]"
                    f"  |  ETA [yellow]{eta_min:.0f} min[/yellow]"
                    f"  |  wall [yellow]{wall_elapsed:.1f} min[/yellow]"
                    f"  |  errors [{'red' if errors else 'white'}]{len(errors)}[/]"
                    f"[/dim]"
                )

    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[yellow bold]⚠  Interrupted by user — progress saved.[/yellow bold]")
        console.print("[yellow]Re-run the script to resume from where it stopped.[/yellow]")

    finally:
        progress.stop()
        ffmpeg_proc.stdout.close()
        ffmpeg_proc.wait()

    # ── summary ──────────────────────────────────────────────────────────────
    total_wall = time.time() - wall_start
    avg_all = sum(frame_times) / len(frame_times) if frame_times else 0

    summary = Table(box=box.ROUNDED, border_style="bright_green", show_header=False, padding=(0, 1))
    summary.add_column("", style="cyan bold")
    summary.add_column("", style="white")
    summary.add_row("Frames in video",   str(frame_idx))
    summary.add_row("Newly enhanced",    str(processed_count))
    summary.add_row("Skipped (existed)", str(skipped_count))
    summary.add_row("Errors",            f"[red]{len(errors)}[/red]" if errors else "[green]0[/green]")
    summary.add_row("Avg time / frame",  f"{avg_all:.2f}s" if avg_all else "—")
    summary.add_row("Peak speed",        f"{60/min(frame_times):.1f} fr/min" if frame_times else "—")
    summary.add_row("Wall time",         f"{total_wall/60:.1f} min")
    console.print(Panel(summary, title="[bold bright_green]Run Summary[/bold bright_green]",
                        border_style="bright_green"))

    if errors:
        console.print(f"[red]Failed frame indices: {[e[0] for e in errors]}[/red]")

    if interrupted:
        console.print("\n[yellow]Skipping reassembly (incomplete). Re-run to finish.[/yellow]")
        sys.exit(0)

    # ── reassemble ───────────────────────────────────────────────────────────
    enhanced_frames = sorted(ENHANCED_DIR.glob("frame_*.jpg"))
    console.print(f"\n[bold]{len(enhanced_frames)} enhanced frames ready.[/bold]")

    if enhanced_frames:
        reassemble_video(fps, has_audio)
    else:
        console.print("[yellow]No enhanced frames found — skipping reassembly.[/yellow]")

    console.rule("[bold bright_green]All done![/bold bright_green]")


if __name__ == "__main__":
    main()

