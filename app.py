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
        height = round(1024 / ar / 8) * 8
    else:
        height = 1024
        width = round(1024 * ar / 8) * 8
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
    return create_dlss5_comparison(image, result), seed


css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

.gradio-container { background: #0a0a0a !important; max-width: 1100px !important; }
#input-img { max-height: 300px; }
#input-img img { max-height: 300px; object-fit: contain; }
#output-img { min-height: 400px; }
#output-img img { max-height: none; width: 100%; height: auto; }
.main-title h1 {
    text-align: center;
    font-family: 'Press Start 2P', monospace !important;
    color: #76B900 !important;
    font-size: 2.2em !important;
    text-shadow: 0 0 10px #76B900, 0 0 30px #76B90066;
    letter-spacing: 2px;
}
.subtitle p {
    text-align: center;
    color: #76B900aa !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55em !important;
    line-height: 1.8 !important;
}
#go-btn {
    background: #76B900 !important;
    color: black !important;
    font-weight: bold;
    font-size: 1.1em;
    min-height: 55px;
    font-family: 'Press Start 2P', monospace !important;
    text-shadow: none;
    border: 2px solid #9aff00 !important;
    box-shadow: 0 0 12px #76B90066;
}
#go-btn:hover { box-shadow: 0 0 25px #76B900; }
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
)) as demo:
    gr.Markdown("# DLSS 5 Anything", elem_classes="main-title")
    gr.Markdown(
        "Turn any image into a DLSS 5 meme",
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

    go_btn.click(
        fn=process,
        inputs=[input_image, prompt, seed, randomize_seed, num_inference_steps],
        outputs=[output_image, seed],
    )

demo.launch()
