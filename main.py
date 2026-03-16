from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import requests
import json
import os
import traceback

# --------------------------------
# Load environment variables
# --------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GEMINI_API_KEY:
    raise Exception("Missing GEMINI_API_KEY")

if not HF_TOKEN:
    raise Exception("Missing HF_TOKEN")

# --------------------------------
# Flask setup
# --------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------
# Gemini client
# --------------------------------

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --------------------------------
# HuggingFace API
# --------------------------------

HF_MODEL_URL = "https://api-inference.huggingface.co/models/zhreyu/ComicStrips-Lora-Fluxdev"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# --------------------------------
# Dialogue Bubble
# --------------------------------

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

        bbox = draw.textbbox((0, 0), test, font=font)

        if bbox[2] < bubble_w - 30:
            line = test
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    y = bubble_y + 20

    for l in lines:
        draw.text((bubble_x + 15, y), l, fill="black", font=font)
        y += 32

    return image


# --------------------------------
# Generate comic panels (Gemini)
# --------------------------------

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
        text = text.split("```")[1].replace("json", "").strip()

    data = json.loads(text)

    return data["panels"][:4]


# --------------------------------
# Generate images via HuggingFace
# --------------------------------

def generate_panel_images(panels):

    images = []

    for panel in panels:

        prompt = f"""
Calvin and Hobbs comic strip,
{panel['scene']}
"""

        payload = {"inputs": prompt}

        response = requests.post(
            HF_MODEL_URL,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"HuggingFace error: {response.text}")

        image = Image.open(BytesIO(response.content))

        image = add_dialogue(image, panel["dialogue"])

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        images.append(img_base64)

    return images


# --------------------------------
# API endpoint
# --------------------------------

@app.route("/output", methods=["POST"])
def generate_comic():

    try:

        data = request.json
        story = data.get("text", "")

        if not story:
            return jsonify({"error": "Story required"}), 400

        panels = generate_story_panels(story)

        images = generate_panel_images(panels)

        return jsonify({
            "status": "success",
            "panels": panels,
            "images": images
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# --------------------------------
# Health check
# --------------------------------

@app.route("/")
def home():
    return "Comic API running"


# --------------------------------
# Run server (Render compatible)
# --------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
