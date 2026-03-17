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

# Optimized prompt
FIXED_ART_PROMPT = (
    "ultra-detailed animated movie still, cinematic lighting, "
    "dynamic pose, vibrant colors, soft lighting, high quality"
)

# ----------------------------
# Dialogue Bubble (Stable)
# ----------------------------
def add_dialogue_bubble(image: Image.Image, dialogue: str) -> Image.Image:
    if not dialogue:
        return image

    draw = ImageDraw.Draw(image)

    # Safe font (works on Vercel)
    font = ImageFont.load_default()

    words = dialogue.split()
    lines = []
    current = ""

    max_width = 500

    for word in words:
        test = f"{current} {word}".strip() if current else word
        bbox = draw.textbbox((0, 0), test, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    padding = 20
    line_height = 25

    bubble_width = max(
        draw.textbbox((0, 0), line, font=font)[2] for line in lines
    ) + padding * 2

    bubble_height = len(lines) * line_height + padding * 2

    bubble_x, bubble_y = 20, 20

    draw.rectangle(
        [
            bubble_x,
            bubble_y,
            bubble_x + bubble_width,
            bubble_y + bubble_height,
        ],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=2,
    )

    y = bubble_y + padding
    for line in lines:
        draw.text((bubble_x + padding, y), line, fill="black", font=font)
        y += line_height

    return image


# ----------------------------
# Gemini Story Generation
# ----------------------------
def generate_story_panels(story: str):
    prompt = f"""
Create a 2-panel cinematic animated story.

Return JSON ONLY:
{{
 "panels":[
  {{"scene":"description","dialogue":"text"}},
  {{"scene":"description","dialogue":"text"}}
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

        return data["panels"], None

    except Exception as e:
        print("GEMINI ERROR:", e)
        return None, str(e)


# ----------------------------
# Image Generation (LIGHTWEIGHT)
# ----------------------------
def generate_panel_images(panels: list):
    images = []

    try:
        for panel in panels:
            prompt = f"{FIXED_ART_PROMPT}, {panel.get('scene','')}"

            image = hf_client.text_to_image(
                prompt,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=512,   # reduced
                height=512,
                guidance_scale=10.0,
                num_inference_steps=30,  # reduced
            )

            image = add_dialogue_bubble(image, panel.get("dialogue", ""))

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
