from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from google import genai
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import json
import os
import logging
import traceback
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# -------------------------------
# Dialogue Bubble
# -------------------------------
def add_dialogue(image: Image.Image, text: str):

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    bubble_x = 30
    bubble_y = 30
    bubble_width = 450
    bubble_height = 200

    draw.rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
        fill="white",
        outline="black",
        width=4,
    )

    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)

        if bbox[2] < bubble_width - 20:
            line = test_line
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


# -------------------------------
# Generate Story Panels
# -------------------------------
def generate_story_panels(story):

    prompt = f"""
You are a professional comic book writer.

Create a coherent 4 panel comic.

Rules:
- same characters in all panels
- cinematic storytelling
- dialogue max 2 sentences
- comic style narrative

Return ONLY JSON:

{{
 "panels":[
  {{"scene":"visual scene description","dialogue":"dialogue"}},
  {{"scene":"visual scene description","dialogue":"dialogue"}},
  {{"scene":"visual scene description","dialogue":"dialogue"}},
  {{"scene":"visual scene description","dialogue":"dialogue"}}
 ]
}}

Story idea:
{story}
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        data = json.loads(text)

        return data["panels"], None

    except Exception as e:

        return None, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# -------------------------------
# Generate Single Comic Panel
# -------------------------------
def generate_single_image(panel):

    prompt = f"""
graphic novel comic panel,
professional comic book illustration,
bold ink lines,
dramatic shadows,
cinematic lighting,
highly detailed artwork,
consistent characters,
{panel['scene']}
"""

    try:

        image = hf_client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=512,
            height=512
        )

        image = add_dialogue(image, panel["dialogue"])

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode(), None

    except Exception as e:

        return None, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# -------------------------------
# Generate All Images
# -------------------------------
def generate_images(panels):

    images = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(generate_single_image, panel)
            for panel in panels
        ]

        for future in concurrent.futures.as_completed(futures):

            img, err = future.result()

            if err:
                return None, err

            images.append(img)

    return images, None


# -------------------------------
# API Endpoint
# -------------------------------
@app.route("/output", methods=["POST"])
def generate_comic():

    try:

        data = request.json
        story = data.get("text", "")

        if not story:
            return jsonify({"error": "Story text missing"}), 400

        panels, err = generate_story_panels(story)

        if err:
            return jsonify(err), 500

        images, err = generate_images(panels)

        if err:
            return jsonify(err), 500

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


# -------------------------------
# Health Check
# -------------------------------
@app.route("/")
def health():

    return {
        "status": "running",
        "huggingface_loaded": bool(HF_API_KEY),
        "gemini_loaded": bool(GEMINI_API_KEY)
    }


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port)
