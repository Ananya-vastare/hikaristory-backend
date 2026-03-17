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

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_API_KEY or not GEMINI_API_KEY:
    raise Exception("Missing API keys in environment variables")

app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 🎬 CINEMATIC REALISTIC STYLE (NO CARTOON)
FIXED_ART_PROMPT = (
    "cinematic film still, ultra realistic, hollywood movie scene, "
    "dramatic lighting, depth of field, 35mm photography, sharp focus, "
    "high detail skin texture, realistic environment, volumetric lighting, "
    "award winning photography, not cartoon, not anime"
)

# ----------------------------
# 🎬 Subtitle Style Dialogue (Clean & Cinematic)
# ----------------------------
def add_dialogue_subtitle(image: Image.Image, dialogue: str) -> Image.Image:
    if not dialogue:
        return image

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x = 20
    y = image.height - 40

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
# Gemini Story Generation (4 panels)
# ----------------------------
def generate_story_panels(story: str):
    prompt = f"""
Create a 4-panel cinematic realistic story.

Rules:
- Same characters across all panels
- Each panel progresses logically
- Each panel must include:
    scene + dialogue

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

        # Extract JSON safely
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise Exception("Invalid JSON from Gemini")

        data = json.loads(match.group(0))

        return data["panels"][:4], None

    except Exception as e:
        print("GEMINI ERROR:", e)
        return None, str(e)


# ----------------------------
# Image Generation (500x500, optimized)
# ----------------------------
def generate_panel_images(panels: list):
    images = []

    try:
        for panel in panels:
            prompt = (
                f"{FIXED_ART_PROMPT}, realistic humans, natural lighting, "
                f"{panel.get('scene','')}"
            )

            image = hf_client.text_to_image(
                prompt,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=500,
                height=500,
                guidance_scale=12.0,
                num_inference_steps=40,
            )

            # Add subtitle (not comic bubble)
            image = add_dialogue_subtitle(
                image, panel.get("dialogue", "")
            )

            buffer = BytesIO()
            image.save(buffer, format="PNG")

            images.append(
                base64.b64encode(buffer.getvalue()).decode()
            )

        return images, None

    except Exception as e:
        print("HF ERROR:", e)
        return None, str(e)


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"})


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

        panels, gem_error = generate_story_panels(story)
        if gem_error:
            return jsonify({
                "status": "error",
                "source": "Gemini",
                "error": gem_error
            }), 500

        images, hf_error = generate_panel_images(panels)
        if hf_error:
            return jsonify({
                "status": "error",
                "source": "HuggingFace",
                "error": hf_error
            }), 500

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
        print("SERVER ERROR:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ----------------------------
# Local Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
