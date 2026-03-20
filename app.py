import os

HF_TOKEN = os.environ.get("HF_TOKEN")

import subprocess
import spaces


def apply_patch():
    import diffusers

    site_packages = os.path.dirname(diffusers.__file__)
    patch_file = os.path.join(os.path.dirname(__file__), "flux2_klein_kv.patch")

    if os.path.exists(patch_file):
        result = subprocess.run(
            ["patch", "-p2", "--forward", "--batch"],
            cwd=os.path.dirname(site_packages),
            stdin=open(patch_file),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Patch applied successfully")
        else:
            print(f"Patch output: {result.stdout}\n{result.stderr}")


apply_patch()

import random
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inter-Bold.ttf")
MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-kv"

print("Loading FLUX.2 Klein 9B KV model...")
pipe = Flux2KleinKVPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=HF_TOKEN)
pipe.to("cuda")
print("Model loaded!")


def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default(size=size)


def create_dlss5_comparison(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    w, h = original.size
    enhanced = enhanced.resize((w, h), Image.LANCZOS)

    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(original, (0, 0))
    canvas.paste(enhanced, (w, 0))

    overlay = Image.new("RGBA", (w * 2, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, int(h * 0.076))
    font = get_font(font_size)

    pad_x = int(font_size * 1.0)
    pad_y = int(font_size * 0.55)

    def draw_label(text, center_x, bottom_y, dark=True, green_bar=False):
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        lw = tw + 2 * pad_x
        lh = th + 2 * pad_y
        gh = max(4, int(lh * 0.13)) if green_bar else 0

        x = center_x - lw // 2
        y = bottom_y - lh - gh

        if dark:
            draw.rectangle(
                [x, y, x + lw, y + lh],
                fill=(10, 10, 10, 225),
                outline=(75, 75, 75, 255),
                width=1,
            )
            draw.text(
                (x + lw // 2, y + lh // 2),
                text,
                fill=(255, 255, 255, 255),
                font=font,
                anchor="mm",
            )
        else:
            draw.rectangle(
                [x, y, x + lw, y + lh],
                fill=(255, 255, 255, 255),
                outline=(190, 190, 190, 255),
                width=1,
            )
            draw.text(
                (x + lw // 2, y + lh // 2),
                text,
                fill=(0, 0, 0, 255),
                font=font,
                anchor="mm",
            )

        if green_bar:
            draw.rectangle(
                [x, y + lh, x + lw, y + lh + gh], fill=(118, 185, 0, 255)
            )

    margin_bottom = int(h * 0.06)
    draw_label("DLSS 5 Off", w // 2, h - margin_bottom, dark=True)
    draw_label(
        "DLSS 5 On", w + w // 2, h - margin_bottom, dark=False, green_bar=True
    )

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    return canvas.convert("RGB")


@spaces.GPU(duration=85)
def process(
    image,
    prompt,
    seed=42,
    randomize_seed=True,
    num_inference_steps=4,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an image!")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Match aspect ratio, max 1024, multiple of 8
    iw, ih = image.size
    ar = iw / ih
    if ar >= 1:
        width = 1024
        height = round(1024 / ar / 16) * 16
    else:
        height = 1024
        width = round(1024 * ar / 16) * 16
    width = max(256, min(1024, width))
    height = max(256, min(1024, height))

    generator = torch.Generator(device=device).manual_seed(seed)

    progress(0.2, desc="Generating DLSS 5 version...")
    result = pipe(
        prompt=prompt,
        image=[image],
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    progress(0.9, desc="Creating comparison...")
    comparison = create_dlss5_comparison(image, result)

    # Resize both to same size for video generation later
    w, h = result.size
    original_resized = image.resize((w, h), Image.LANCZOS)

    return comparison, seed, original_resized, result


import tempfile


def create_slider_video(original, enhanced, fps=30, pause_duration=0.8, slide_duration=3.0, hold_duration=1.0):
    """Create a DLSS 5 slider comparison video from raw original + enhanced images."""
    w, h = original.size
    if w % 2: w -= 1
    if h % 2: h -= 1
    original = original.resize((w, h)).convert("RGBA")
    enhanced = enhanced.resize((w, h)).convert("RGBA")

    font_size = max(20, int(h * 0.076))
    font = get_font(font_size)
    pad_x = int(font_size * 1.0)
    pad_y = int(font_size * 0.55)
    margin = int(h * 0.06)

    # Same centered label position for both — slider reveals one, hides the other
    bbox_off = font.getbbox("DLSS 5 Off")
    bbox_on = font.getbbox("DLSS 5 On")
    tw = max(bbox_off[2] - bbox_off[0], bbox_on[2] - bbox_on[0])
    th = max(bbox_off[3] - bbox_off[1], bbox_on[3] - bbox_on[1])
    lw = tw + 2 * pad_x
    lh = th + 2 * pad_y
    green_h = max(4, int(lh * 0.13))

    lx1 = (w - lw) // 2
    lx2 = lx1 + lw
    ly2 = h - margin - green_h
    ly1 = ly2 - lh

    # Bake "DLSS 5 Off" into original
    off_ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(off_ov)
    od.rectangle([lx1, ly1, lx2, ly2], fill=(10, 10, 10, 225), outline=(75, 75, 75, 255))
    od.text(((lx1 + lx2) // 2, (ly1 + ly2) // 2), "DLSS 5 Off",
             fill=(255, 255, 255, 255), font=font, anchor="mm")
    original = Image.alpha_composite(original, off_ov)

    # Bake "DLSS 5 On" into enhanced
    on_ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    nd = ImageDraw.Draw(on_ov)
    nd.rectangle([lx1, ly1, lx2, ly2], fill=(255, 255, 255, 255), outline=(190, 190, 190, 255))
    nd.text(((lx1 + lx2) // 2, (ly1 + ly2) // 2), "DLSS 5 On",
             fill=(0, 0, 0, 255), font=font, anchor="mm")
    nd.rectangle([lx1, ly2, lx2, ly2 + green_h], fill=(118, 185, 0, 255))
    enhanced = Image.alpha_composite(enhanced, on_ov)

    pause_frames = int(fps * pause_duration)
    slide_frames = int(fps * slide_duration)
    hold_frames = int(fps * hold_duration)
    total_frames = pause_frames + slide_frames + hold_frames

    # Convert to numpy for fast compositing
    import numpy as np
    orig_arr = np.array(original.convert("RGB"))
    enh_arr = np.array(enhanced.convert("RGB"))

    # Pipe raw frames directly to ffmpeg (no disk I/O)
    output_path = tempfile.mktemp(suffix=".mp4")
    ffmpeg_proc = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast", output_path
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mid_y = h // 2

    for i in range(total_frames):
        if i < pause_frames:
            pos = 0.0
        elif i < pause_frames + slide_frames:
            t = (i - pause_frames) / slide_frames
            pos = t * t * (3 - 2 * t)
        else:
            pos = 1.0

        slider_x = int(w * pos)

        # Composite: left of slider = enhanced (with On label), right = original (with Off label)
        frame_arr = orig_arr.copy()
        if slider_x > 0:
            frame_arr[:, :slider_x] = enh_arr[:, :slider_x]

        # Draw slider line directly in numpy
        if 0 < slider_x < w:
            x1 = max(0, slider_x - 2)
            x2 = min(w, slider_x + 3)
            frame_arr[:, x1:x2] = 255
            # Diamond handle
            size = 14
            for dy in range(-size, size + 1):
                half = size - abs(dy)
                dx1 = max(0, slider_x - half)
                dx2 = min(w, slider_x + half + 1)
                frame_arr[mid_y + dy, dx1:dx2] = 255

        ffmpeg_proc.stdin.write(frame_arr.tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")

    return output_path


def generate_video(original_state, enhanced_state):
    if original_state is None or enhanced_state is None:
        raise gr.Error("Generate a DLSS 5 comparison first!")
    return create_slider_video(original_state, enhanced_state)


css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

.gradio-container { background: #0a0a0a !important; max-width: 900px !important; margin: 0 auto !important; padding-top: 30px !important; }
#input-img { max-height: 300px; }
#input-img img { max-height: 300px; object-fit: contain; }
.gallery-item { border-color: #333 !important; }
.progress-bar { background-color: #76B900 !important; }
.progress-bar-wrap { background-color: #1a1a1a !important; border-color: #333 !important; }
.eta-bar { background-color: #76B90033 !important; }
.label-wrap { background: #111111 !important; color: #cccccc !important; }
.gallery-item { background: #111111 !important; }
* { --neutral-50: #1a1a1a !important; --neutral-100: #222222 !important; --neutral-200: #333333 !important; }
.upload-area { border-color: #76B90044 !important; }
.block { border-color: #76B90033 !important; }
#output-img .image-container { height: auto !important; }
#output-img .image-frame { height: auto !important; }
#output-img .image-frame img { width: 100% !important; height: auto !important; max-height: none !important; object-fit: contain !important; }
.main-title h1 {
    text-align: center;
    font-family: 'Press Start 2P', monospace !important;
    color: #9aff00 !important;
    font-size: 2.2em !important;
    text-shadow: 0 0 15px #9aff00, 0 0 40px #76B900, 0 0 80px #76B90044;
    letter-spacing: 2px;
}
.subtitle p {
    text-align: center;
    color: #9aff00cc !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55em !important;
    line-height: 1.8 !important;
    text-shadow: 0 0 8px #76B90066;
}
#go-btn {
    background: linear-gradient(135deg, #76B900, #9aff00) !important;
    color: black !important;
    font-weight: bold;
    font-size: 1.2em;
    min-height: 60px;
    font-family: 'Press Start 2P', monospace !important;
    text-shadow: none;
    border: 2px solid #9aff00 !important;
    box-shadow: 0 0 20px #76B900aa, inset 0 0 10px #ffffff22;
}
#go-btn:hover { box-shadow: 0 0 35px #76B900, inset 0 0 15px #ffffff33; }
#video-btn {
    background: #2563eb !important;
    color: white !important;
    font-weight: bold;
    font-size: 1em;
    min-height: 50px;
    font-family: 'Press Start 2P', monospace !important;
    border: 2px solid #3b82f6 !important;
    box-shadow: 0 0 12px #2563eb66;
}
#video-btn:hover { box-shadow: 0 0 25px #2563eb; }
#video-btn.loading::before {
    content: '';
    display: inline-block;
    width: 1em; height: 1em;
    border: 3px solid #ffffff44;
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 0.5em;
    vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }
#video-download { border: none !important; background: transparent !important; padding: 0 !important; }
.dark { --body-background-fill: #0a0a0a; }
"""

with gr.Blocks(title="DLSS 5 Anything", css=css, theme=gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Press Start 2P"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    block_border_color="#76B90033",
    block_border_color_dark="#76B90033",
    block_label_text_color="#76B900",
    block_label_text_color_dark="#76B900",
    block_title_text_color="#76B900",
    block_title_text_color_dark="#76B900",
    body_text_color="#cccccc",
    body_text_color_dark="#cccccc",
    button_primary_background_fill="#76B900",
    button_primary_background_fill_dark="#76B900",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    input_background_fill="#1a1a1a",
    input_background_fill_dark="#1a1a1a",
    input_border_color="#76B90044",
    input_border_color_dark="#76B90044",
    border_color_accent="#76B900",
    border_color_accent_dark="#76B900",
    border_color_primary="#333333",
    border_color_primary_dark="#333333",
    background_fill_secondary="#111111",
    background_fill_secondary_dark="#111111",
    table_border_color="#333333",
    table_border_color_dark="#333333",
    background_fill_primary="#0f0f0f",
    background_fill_primary_dark="#0f0f0f",
    shadow_drop="none",
    shadow_drop_lg="none",
    checkbox_background_color="#1a1a1a",
    checkbox_background_color_dark="#1a1a1a",
    checkbox_background_color_selected="#76B900",
    checkbox_background_color_selected_dark="#76B900",
    slider_color="#76B900",
    slider_color_dark="#76B900",
)) as demo:
    gr.Markdown("# DLSS 5 Anything", elem_classes="main-title")
    gr.Markdown(
        "Turn any image into a DLSS 5 meme (using FLUX.2-klein-9b-kv)",
        elem_classes="subtitle",
    )

    prompt = gr.Textbox(value="make it more realistic", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil", elem_id="input-img")

        with gr.Column(scale=1):
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Inference steps", minimum=1, maximum=20, step=1, value=4
                )
            go_btn = gr.Button("DLSS 5 it!", elem_id="go-btn", variant="primary")

            gr.Examples(
                examples=[
                    ["example_mario.png"],
                    ["example_oblivion.jpg"],
                    ["example_gta_sa.jpeg"],
                    ["example_roland.jpg"],
                    ["example_geralt.png"],
                ],
                inputs=[input_image],
            )

    output_image = gr.Image(label="Result", type="pil", elem_id="output-img")

    # Hidden state for video generation
    original_state = gr.State(None)
    enhanced_state = gr.State(None)

    video_btn = gr.Button("Generate & download video", elem_id="video-btn", visible=False)
    video_file = gr.File(visible=False, elem_id="video-download")

    def on_generate(image, prompt, seed, randomize_seed, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
        comparison, seed, orig, enh = process(image, prompt, seed, randomize_seed, num_inference_steps, progress)
        return comparison, seed, orig, enh, gr.update(visible=True), gr.update(visible=False)

    go_btn.click(
        fn=on_generate,
        inputs=[input_image, prompt, seed, randomize_seed, num_inference_steps],
        outputs=[output_image, seed, original_state, enhanced_state, video_btn, video_file],
    )

    # Hide video button when input image changes
    input_image.change(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False), None, None),
        inputs=[],
        outputs=[video_btn, video_file, original_state, enhanced_state],
    )

    def make_video(orig, enh):
        if orig is None or enh is None:
            raise gr.Error("Generate a DLSS 5 comparison first!")
        path = create_slider_video(orig, enh)
        return gr.update(value=path, visible=True)

    video_btn.click(
        fn=lambda: gr.update(value="⏳ Generating video...", interactive=False),
        inputs=[],
        outputs=[video_btn],
    ).then(
        fn=make_video,
        inputs=[original_state, enhanced_state],
        outputs=[video_file],
    ).then(
        fn=lambda: gr.update(value="Generate & download video", interactive=True),
        inputs=[],
        outputs=[video_btn],
    )

demo.launch()
