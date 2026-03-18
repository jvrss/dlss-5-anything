"""Test script for slider video — imports from app.py's create_slider_video."""
from PIL import Image, ImageEnhance

original = Image.open("/tmp/game_screenshots/11_oblivion_face1.jpg")
enhanced = ImageEnhance.Color(original).enhance(1.3)
enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

print(f"Original: {original.size}")
print("Generating slider video...")

# Import from app
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app import create_slider_video
create_slider_video(original, enhanced, output_path="/tmp/dlss5_slider.mp4")
print("Done! /tmp/dlss5_slider.mp4")
