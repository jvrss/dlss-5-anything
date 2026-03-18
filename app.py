import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
import io
import os

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inter-Bold.ttf")
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

hf_client = InferenceClient(token=os.environ.get("HF_TOKEN"))


def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default(size=size)


def create_dlss5_comparison(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    w, h = original.size
    enhanced = enhanced.resize((w, h), Image.LANCZOS)

    # Side-by-side canvas
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(original, (0, 0))
    canvas.paste(enhanced, (w, 0))

    # RGBA overlay for semi-transparent labels
    overlay = Image.new("RGBA", (w * 2, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, int(h * 0.038))
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


def process(image, prompt):
    if image is None:
        raise gr.Error("Please upload an image!")

    try:
        enhanced = hf_client.image_to_image(
            image=image,
            prompt=prompt,
            model=MODEL_ID,
        )
        return create_dlss5_comparison(image, enhanced)
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")


css = """
.main-title { text-align: center; }
.subtitle { text-align: center; color: #888; font-size: 1.05em; }
#go-btn { background: #76B900 !important; color: white !important; font-weight: bold; font-size: 1.1em; min-height: 50px; }
"""

with gr.Blocks(title="DLSS 5 Anything", css=css) as demo:
    gr.Markdown("# DLSS 5 Anything", elem_classes="main-title")
    gr.Markdown(
        "Turn any image into a DLSS 5 meme — powered by FLUX.2 Klein",
        elem_classes="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil")
            prompt = gr.Textbox(
                label="Prompt",
                value="make it more realistic",
                placeholder="e.g. make it more realistic",
            )
            go_btn = gr.Button("DLSS 5 it!", elem_id="go-btn", variant="primary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Result", type="pil")

    go_btn.click(fn=process, inputs=[input_image, prompt], outputs=[output_image])

demo.launch()
