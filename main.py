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
from dotenv import load_dotenv

# ================== SETUP ==================
load_dotenv()
logging.basicConfig(level=logging.INFO)

HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def add_dialogue(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    # Bubble position (top area like comics)
    padding = 20
    max_width = image.width - 200

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)

        if bbox[2] < max_width:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    text_height = len(lines) * 40 + padding * 2

    # Speech bubble (top)
    box = [
        100,
        50,
        image.width - 100,
        50 + text_height,
    ]

    # Bubble
    draw.ellipse(box, fill="white", outline="black", width=3)

    # Tail (speech pointer)
    draw.polygon(
        [(box[0] + 100, box[3]), (box[0] + 150, box[3]), (box[0] + 120, box[3] + 40)],
        fill="white",
        outline="black",
    )

    # Text
    y = box[1] + padding
    for line in lines:
        draw.text((box[0] + padding, y), line, fill="black", font=font)
        y += 40

    return image


def generate_story(story_text):
    if not gemini_client:
        return None, "Gemini API key missing"

    prompt = f"""
Create a 4-panel comic.

Rules:
- Same character in all panels
- Describe character in EACH panel
- Cinematic scenes
- Dialogue = 1 short sentence
- Return ONLY JSON

Format:
{{
 "panels":[
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}}
 ]
}}

Story: {story_text}
"""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        text = res.text.strip()

        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        data = json.loads(text)
        return data["panels"], None

    except Exception as e:
        logging.error(f"Gemini error: {str(e)}")
        return None, "Story generation failed"


# ================== IMAGE ==================
def generate_image(panel, style):

    if not hf_client:
        return None, "HuggingFace API key missing"

    STYLE_MAP = {
        "cartoonish": "modern comic, bold outlines, vibrant cinematic lighting",
        "soft": "watercolor illustration, pastel tones",
        "dramatic": "cinematic lighting, high contrast shadows",
        "manga": "black and white manga panel, screentones",
    }

    prompt = f"""
{STYLE_MAP.get(style, STYLE_MAP['cartoonish'])},
highly detailed, professional art,
consistent character design,
{panel['scene']}
"""

    try:
        image = hf_client.text_to_image(
            prompt=prompt,
            negative_prompt="blurry, distorted, bad anatomy",
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=1024,
            height=1024,
        )

    except Exception as e:
        logging.warning("Turbo failed → fallback SDXL")

        try:
            image = hf_client.text_to_image(
                prompt=prompt,
                negative_prompt="blurry, distorted",
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=600,
                height=600,
            )
        except Exception as e2:
            logging.error(f"HF error: {str(e2)}")
            return None, "Image generation failed"

    try:
        image = add_dialogue(image, panel["dialogue"])
    except:
        pass

    return encode_image(image), None


# ================== PIPELINE ==================
def generate_comic_pipeline(story, style):
    panels, err = generate_story(story)
    if err:
        return None, err

    images = []

    for panel in panels:
        logging.info(f"Generating: {panel['scene'][:40]}")

        img, err = generate_image(panel, style)
        if err:
            return None, err

        images.append(img)

    return {"panels": panels, "images": images}, None


# ================== ROUTES ==================
@app.route("/output", methods=["POST"])
def output():
    try:
        data = request.json
        story = data.get("text", "")
        style = data.get("image_style", "cartoonish")

        if not story:
            return jsonify({"status": "error", "message": "No story provided"}), 400

        result, err = generate_comic_pipeline(story, style)

        if err:
            return jsonify({"status": "error", "message": err}), 500

        return jsonify({"status": "success", **result})

    except Exception as e:
        logging.exception("SERVER ERROR")

        return jsonify({"status": "error", "message": "Internal server error"}), 500


@app.route("/")
def health():
    return {
        "status": "running",
        "hf_loaded": bool(HF_API_KEY),
        "gemini_loaded": bool(GEMINI_API_KEY),
    }


# ================== RUN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
