# title: DLSS 5 Anything
emoji: 🎮 \
colorFrom: green \
colorTo: gray \
sdk: gradio \
sdk_version: 5.23.0 \
app_file: app.py \
pinned: false \
short_description: Turn any image into a DLSS 5 meme (using FLUX.2-klein-9b-kv) 

Setar ambiente:
py -3.11 -m venv venv

Configurar cuda
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
