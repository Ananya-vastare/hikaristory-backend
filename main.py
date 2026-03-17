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
    raise Exception("❌ Missing API keys in environment variables")

# ----------------------------
# 🚀 App Init
# ----------------------------
app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------
# 🎨 COMIC STYLE PROMPT (FIXED)
# ----------------------------
COMIC_STYLE_PROMPT = (
    "pop art comic style, retro comic book illustration, "
    "bold black outlines, halftone dots, vibrant colors, "
    "dramatic expression, highly detailed face, "
    "vintage 1960s comic style, Roy Lichtenstein inspired, "
    "flat shading, high contrast, clean lines, "
    "comic panel, dynamic composition, expressive emotions, "
    "pop art explosion background"
)

NEGATIVE_PROMPT = (
    "realistic photo, blurry, low quality, 3d render, dull colors, "
    "bad anatomy, extra limbs, distorted face"
)

# ----------------------------
# 💬 Add Comic Subtitle (Simple)
# ----------------------------
def add_dialogue_subtitle(image: Image.Image, dialogue: str) -> Image.Image:
    if not dialogue:
        return image

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    x = 20
    y = image.height - 60

    draw.text(
        (x, y),
        dialogue,
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    return image


# ----------------------------
# 🧠 Generate Story Panels (Gemini)
# ----------------------------
def generate_story_panels(story: str):
    prompt = f"""
Create a 4-panel comic story.

Rules:
- Same characters across all panels
- Each panel progresses logically
- Keep scenes visually descriptive for image generation
- Each panel must include:
    scene + short dialogue (1 line max)

Return ONLY JSON:
{{
 "panels":[
  {{"scene":"...","dialogue":"..."}},
  {{"scene":"...","dialogue":"..."}},
  {{"scene":"...","dialogue":"..."}},
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
            raise Exception("Invalid JSON from Gemini")

        data = json.loads(match.group(0))

        return data["panels"][:4], None

    except Exception as e:
        print("❌ GEMINI ERROR:", e)
        return None, str(e)


# ----------------------------
# 🎨 Generate Comic Images
# ----------------------------
def generate_panel_images(panels: list):
    images = []

    try:
        for panel in panels:
            prompt = (
                f"{COMIC_STYLE_PROMPT}, "
                f"{panel.get('scene', '')}, "
                "close-up shot, expressive face"
            )

            image = hf_client.text_to_image(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=768,
                height=768,
                guidance_scale=8.5,
                num_inference_steps=50,
            )

            image = add_dialogue_subtitle(
                image,
                panel.get("dialogue", "")
            )

            buffer = BytesIO()
            image.save(buffer, format="PNG")

            images.append(
                base64.b64encode(buffer.getvalue()).decode()
            )

        return images, None

    except Exception as e:
        print("❌ HF ERROR:", e)
        return None, str(e)


# ----------------------------
# 🏠 Health Check
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "🎨 Comic Generator API is live"
    })


# ----------------------------
# 🎬 Main Endpoint
# ----------------------------
@app.route("/output", methods=["POST"])
def generate_comic():
    try:
        data = request.json
        story = data.get("text", "").strip()

        if not story:
            return jsonify({
                "status": "error",
                "message": "No storyline provided"
            }), 400

        # 🧠 Generate story
        panels, gem_error = generate_story_panels(story)
        if gem_error:
            return jsonify({
                "status": "error",
                "source": "Gemini",
                "error": gem_error
            }), 500

        # 🎨 Generate images
        images, hf_error = generate_panel_images(panels)
        if hf_error:
            return jsonify({
                "status": "error",
                "source": "HuggingFace",
                "error": hf_error
            }), 500

        # 📦 Combine results
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
            "panels": result
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ----------------------------
# ▶️ Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
