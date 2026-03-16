from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import json
import os
import traceback
import torch
from diffusers import AutoPipelineForText2Image

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY missing")

# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)
CORS(app)

# -----------------------------
# Gemini Client
# -----------------------------

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# Load Flux + Comic LoRA
# -----------------------------

print("Loading Flux model...")

pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

pipeline = pipeline.to("cuda")

print("Loading Comic LoRA...")

pipeline.load_lora_weights(
    "zhreyu/ComicStrips-Lora-Fluxdev",
    weight_name="ComicStrips_flux_lora_v1_fp16.safetensors"
)

print("Model ready.")


# -----------------------------
# Dialogue Bubble
# -----------------------------

def add_dialogue(image, text):

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    bubble_x = 40
    bubble_y = 40
    bubble_w = 500
    bubble_h = 200

    draw.rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_w, bubble_y + bubble_h],
        fill="white",
        outline="black",
        width=4
    )

    words = text.split()
    lines = []
    line = ""

    for word in words:

        test = f"{line} {word}".strip()

        bbox = draw.textbbox((0,0), test, font=font)

        if bbox[2] < bubble_w - 30:
            line = test
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    y = bubble_y + 20

    for l in lines:
        draw.text((bubble_x+15,y),l,fill="black",font=font)
        y += 32

    return image


# -----------------------------
# Generate Comic Story
# -----------------------------

def generate_story_panels(story):

    prompt = f"""
Create a 4 panel comic story.

Return JSON only.

Format:
{{
 "panels":[
  {{"scene":"visual scene","dialogue":"dialogue"}},
  {{"scene":"visual scene","dialogue":"dialogue"}},
  {{"scene":"visual scene","dialogue":"dialogue"}},
  {{"scene":"visual scene","dialogue":"dialogue"}}
 ]
}}

Story idea:
{story}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if "```" in text:
        text = text.split("```")[1].replace("json","").strip()

    data = json.loads(text)

    return data["panels"][:4]


# -----------------------------
# Generate Images using LoRA
# -----------------------------

def generate_panel_images(panels):

    images = []

    for panel in panels:

        prompt = f"""
Calvin and Hobbs comic strip,
{panel['scene']}
"""

        image = pipeline(
            prompt,
            num_inference_steps=30,
            guidance_scale=4.0,
            height=1024,
            width=1024
        ).images[0]

        image = add_dialogue(image, panel["dialogue"])

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        images.append(img_base64)

    return images


# -----------------------------
# API Endpoint
# -----------------------------

@app.route("/output", methods=["POST"])
def generate_comic():

    try:

        data = request.json
        story = data.get("text","")

        if not story:
            return jsonify({"error":"Story required"}),400

        panels = generate_story_panels(story)

        images = generate_panel_images(panels)

        return jsonify({
            "status":"success",
            "panels":panels,
            "images":images
        })

    except Exception as e:

        return jsonify({
            "error":str(e),
            "traceback":traceback.format_exc()
        }),500


# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
