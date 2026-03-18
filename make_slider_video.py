"""
Generate a DLSS 5 sliding comparison video from original + enhanced images.
"""

from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import subprocess


def get_font(size):
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inter-Bold.ttf")
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default(size=size)


def create_slider_frame(original, enhanced, slider_pos, w, h, font, pad_x, pad_y,
                         off_label_rect, on_label_rect, on_green_h):
    """Create one frame. slider_pos 0.0 (all original) to 1.0 (all enhanced)."""
    slider_x = int(w * slider_pos)

    # Composite: left = enhanced (revealed), right = original
    frame = Image.new("RGBA", (w, h))
    frame.paste(original, (0, 0))
    if slider_x > 0:
        frame.paste(enhanced.crop((0, 0, slider_x, h)), (0, 0))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fixed labels: "DLSS 5 Off" bottom-right, "DLSS 5 On" bottom-left
    # Only show if not occluded by slider
    ox, oy, ox2, oy2 = off_label_rect
    if slider_x < ox:  # slider hasn't reached the Off label yet
        draw.rectangle([ox, oy, ox2, oy2],
                        fill=(10, 10, 10, 225), outline=(75, 75, 75, 255))
        draw.text(((ox + ox2) // 2, (oy + oy2) // 2), "DLSS 5 Off",
                   fill=(255, 255, 255, 255), font=font, anchor="mm")

    nx, ny, nx2, ny2 = on_label_rect
    if slider_x > nx2:  # slider has passed the On label
        draw.rectangle([nx, ny, nx2, ny2],
                        fill=(255, 255, 255, 255), outline=(190, 190, 190, 255))
        draw.text(((nx + nx2) // 2, (ny + ny2) // 2), "DLSS 5 On",
                   fill=(0, 0, 0, 255), font=font, anchor="mm")
        draw.rectangle([nx, ny2, nx2, ny2 + on_green_h],
                        fill=(118, 185, 0, 255))

    # Slider line
    if 0 < slider_x < w:
        draw.rectangle([slider_x - 2, 0, slider_x + 2, h], fill=(255, 255, 255, 230))
        # Diamond handle
        mid_y = h // 2
        size = 14
        for dy in range(-size, size + 1):
            half = size - abs(dy)
            draw.rectangle([slider_x - half, mid_y + dy, slider_x + half, mid_y + dy],
                            fill=(255, 255, 255, 255))

    frame = Image.alpha_composite(frame, overlay)
    return frame.convert("RGB")


def create_slider_video(original, enhanced, output_path, fps=30, slide_duration=3.0, hold_duration=1.0):
    w, h = original.size
    if w % 2: w -= 1
    if h % 2: h -= 1
    original = original.resize((w, h)).convert("RGBA")
    enhanced = enhanced.resize((w, h)).convert("RGBA")

    # Bigger font: 7.6% of height (2x the static comparison labels)
    font_size = max(20, int(h * 0.076))
    font = get_font(font_size)
    pad_x = int(font_size * 1.0)
    pad_y = int(font_size * 0.55)

    # Pre-compute fixed label positions
    margin = int(h * 0.06)

    # "DLSS 5 Off" — fixed bottom-right
    bbox_off = font.getbbox("DLSS 5 Off")
    lw_off = bbox_off[2] - bbox_off[0] + 2 * pad_x
    lh_off = bbox_off[3] - bbox_off[1] + 2 * pad_y
    off_x2 = w - int(w * 0.05)
    off_x1 = off_x2 - lw_off
    off_y2 = h - margin
    off_y1 = off_y2 - lh_off
    off_label_rect = (off_x1, off_y1, off_x2, off_y2)

    # "DLSS 5 On" — fixed bottom-left
    bbox_on = font.getbbox("DLSS 5 On")
    lw_on = bbox_on[2] - bbox_on[0] + 2 * pad_x
    lh_on = bbox_on[3] - bbox_on[1] + 2 * pad_y
    on_green_h = max(4, int(lh_on * 0.13))
    on_x1 = int(w * 0.05)
    on_x2 = on_x1 + lw_on
    on_y2 = h - margin - on_green_h
    on_y1 = on_y2 - lh_on
    on_label_rect = (on_x1, on_y1, on_x2, on_y2)

    slide_frames = int(fps * slide_duration)
    hold_frames = int(fps * hold_duration)
    total_frames = slide_frames + hold_frames

    tmpdir = tempfile.mkdtemp()
    frame_paths = []

    for i in range(total_frames):
        if i < slide_frames:
            t = i / slide_frames
            pos = t * t * (3 - 2 * t)  # smoothstep
        else:
            pos = 1.0

        frame = create_slider_frame(
            original, enhanced, pos, w, h, font, pad_x, pad_y,
            off_label_rect, on_label_rect, on_green_h
        )
        fp = os.path.join(tmpdir, f"frame_{i:04d}.png")
        frame.save(fp)
        frame_paths.append(fp)

        if (i + 1) % 30 == 0 or i == total_frames - 1:
            print(f"  Frame {i+1}/{total_frames}")

    result = subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast", output_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    for fp in frame_paths:
        try: os.unlink(fp)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass

    return output_path


if __name__ == "__main__":
    # Use raw source images (not the comparison which has labels baked in)
    original = Image.open("/tmp/game_screenshots/11_oblivion_face1.jpg")
    # For testing, create a fake "enhanced" by adjusting the original
    # In production, this comes from the model output
    from PIL import ImageEnhance
    enhanced = ImageEnhance.Color(original).enhance(1.3)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)
    # Resize both to same size
    enhanced = enhanced.resize(original.size)
    print(f"Original: {original.size}, Enhanced: {enhanced.size}")
    print("Generating slider video...")
    create_slider_video(original, enhanced, "/tmp/dlss5_slider.mp4")
    print("Done! /tmp/dlss5_slider.mp4")
