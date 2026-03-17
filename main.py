from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from google import genai
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import json
import os
import re

# ----------------------------
# 🔐 Load Environment Variables
# ----------------------------
load_dotenv()

HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_API_KEY or not GEMINI_API_KEY:
    raise Exception("❌ Missing API keys")

# ----------------------------
# 🚀 App Init
# ----------------------------
app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------
# 🎨 IMPROVED IMAGE STYLE
# ----------------------------
STYLE_PROMPT = (
    "cinematic comic illustration, semi-realistic, ultra detailed, "
    "modern graphic novel style, dramatic lighting, sharp focus, "
    "realistic anatomy, consistent character design, "
    "high detail face, volumetric lighting, 4k quality"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, pop art, lichtenstein, blurry, low quality, "
    "bad anatomy, distorted face, extra limbs"
)

# ----------------------------
# 💬 SPEECH BUBBLE FUNCTION
# ----------------------------
def draw_speech_bubble(image: Image.Image, text: str) -> Image.Image:
    if not text:
        return image

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    padding = 20
    max_width = image.width - 120

    # Wrap text
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word
        w, h = draw.textbbox((0, 0), test, font=font)[2:]
        if w < max_width:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    line_height = 28
    text_height = len(lines) * line_height
    text_width = max([draw.textbbox((0,0), l, font=font)[2] for l in lines])

    x = 50
    y = image.height - text_height - 120

    # Bubble box
    box = [
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding
    ]

    draw.rounded_rectangle(box, radius=25, fill="white", outline="black", width=3)

    # Tail
    draw.polygon([
        (x + 60, y + text_height + padding),
        (x + 90, y + text_height + padding),
        (x + 75, y + text_height + padding + 30)
    ], fill="white", outline="black")

    # Draw text
    offset_y = y
    for line in lines:
        draw.text((x, offset_y), line, fill="black", font=font)
        offset_y += line_height

    return image

# ----------------------------
# 🧠 GENERATE STORY PANELS
# ----------------------------
def generate_story(story: str):
    prompt = f"""
Create a 4-panel cinematic comic story.

STRICT RULES:
- ONE main character only
- Character must remain IDENTICAL in all panels
- Define character clearly (face, hair, outfit)
- ONE continuous storyline (no randomness)
- Each panel must logically continue the previous one

Return ONLY JSON:
{{
 "main_character": "detailed character description",
 "theme": "short theme",
 "panels":[
  {{"scene":"...","dialogue":"..."}} ,
  {{"scene":"...","dialogue":"..."}} ,
  {{"scene":"...","dialogue":"..."}} ,
  {{"scene":"...","dialogue":"..."}}
 ]
}}

Story:
{story}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise Exception("Invalid JSON")

        data = json.loads(match.group(0))

        return data, None

    except Exception as e:
        print("❌ Gemini Error:", e)
        return None, str(e)

# ----------------------------
# 🎨 GENERATE IMAGES
# ----------------------------
def generate_images(panels, character):
    images = []

    try:
        for panel in panels:
            prompt = (
                f"{STYLE_PROMPT}, "
                f"{character}, "
                f"{panel['scene']}, "
                "same character, consistent face, same outfit"
            )

            img = hf_client.text_to_image(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=768,
                height=768,
                guidance_scale=8.5,
                num_inference_steps=50,
            )

            img = draw_speech_bubble(img, panel["dialogue"])

            buffer = BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                base64.b64encode(buffer.getvalue()).decode()
            )

        return images, None

    except Exception as e:
        print("❌ Image Error:", e)
        return None, str(e)

# ----------------------------
# 🏠 HEALTH CHECK
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "🔥 Cinematic Comic API Live"
    })

# ----------------------------
# 🎬 MAIN ENDPOINT
# ----------------------------
@app.route("/output", methods=["POST"])
def generate_comic():
    try:
        data = request.json
        story_input = data.get("text", "").strip()

        if not story_input:
            return jsonify({
                "status": "error",
                "message": "No story provided"
            }), 400

        # 🧠 Story generation
        story_data, err = generate_story(story_input)
        if err:
            return jsonify({
                "status": "error",
                "source": "Gemini",
                "error": err
            }), 500

        character = story_data["main_character"]
        panels = story_data["panels"]

        # 🎨 Image generation
        images, err = generate_images(panels, character)
        if err:
            return jsonify({
                "status": "error",
                "source": "HuggingFace",
                "error": err
            }), 500

        # 📦 Final output
        result = [
            {
                "scene": p["scene"],
                "dialogue": p["dialogue"],
                "image": img
            }
            for p, img in zip(panels, images)
        ]

        return jsonify({
            "status": "success",
            "character": character,
            "panels": result
        })

    except Exception as e:
        print("❌ Server Error:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

