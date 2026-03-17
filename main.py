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

# ================== CLIENT INITIALIZATION ==================
hf_client = None
gemini_client = None

if HF_API_KEY:
    try:
        hf_client = InferenceClient(api_key=HF_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Hugging Face client: {e}")

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")

# ================== UTILITIES ==================
def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, current = [], ""
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
    return lines

def add_dialogue(image: Image.Image, dialogue: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    padding = 10
    max_width = image.width - 60
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    lines = wrap_text(draw, dialogue, font, max_width)
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    text_height = len(lines) * (line_height + 3) + padding * 2

    bubble_x0, bubble_y0 = 30, 30
    bubble_x1 = bubble_x0 + max([draw.textbbox((0, 0), line, font=font)[2] for line in lines]) + padding*2
    bubble_y1 = bubble_y0 + text_height

    # Draw bubble
    draw.rounded_rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1],
                           radius=15, fill="white", outline="black", width=2)

    # Draw bubble tail
    tail_width, tail_height = 25, 15
    tail_x0 = bubble_x0 + bubble_x1 // 4
    tail_y0 = bubble_y1
    draw.polygon([(tail_x0, tail_y0),
                  (tail_x0 + tail_width, tail_y0),
                  (tail_x0 + tail_width // 2, tail_y0 + tail_height)],
                 fill="white", outline="black")

    # Draw text
    y = bubble_y0 + padding
    for line in lines:
        draw.text((bubble_x0 + padding, y), line, fill="black", font=font)
        y += line_height + 3

    return image

# ================== STORY GENERATION ==================
def generate_story(story_text):
    if not gemini_client:
        raise RuntimeError("Gemini client not initialized or GEMINI_API_KEY missing")

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
   {{"scene":"...", "dialogue":"..."}} x4
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
        if "panels" not in data or not isinstance(data["panels"], list):
            raise ValueError("Invalid response structure from Gemini")
        return data["panels"]
    except Exception as e:
        logging.error(f"Gemini story generation failed: {e}")
        raise RuntimeError(f"Story generation failed: {str(e)}") from e

# ================== IMAGE GENERATION ==================
def generate_image(panel, style):
    if not hf_client:
        return {"error": "Hugging Face client not initialized or ACCESS_TOKEN missing"}, 500

    width, height = 768, 512
    STYLE_MAP = {
        "cartoonish": "modern comic, bold outlines, vibrant cinematic lighting, highly detailed",
        "soft": "watercolor illustration, pastel tones, highly detailed",
        "dramatic": "cinematic lighting, high contrast shadows, highly detailed",
        "manga": "black and white manga panel, screentones, highly detailed",
    }

    prompt = f"""
{STYLE_MAP.get(style, STYLE_MAP['cartoonish'])},
professional, cinematic, consistent character design,
{panel.get('scene', '')}
"""
    try:
        image_bytes = hf_client.text_to_image(
            prompt=prompt,
            negative_prompt="blurry, distorted, bad anatomy",
            model="stabilityai/stable-diffusion-xl-refiner-1.0",
            width=width,
            height=height,
        )
        if isinstance(image_bytes, bytes):
            image = Image.open(BytesIO(image_bytes))
        else:
            raise ValueError("Unexpected Hugging Face response format")
    except Exception as e:
        logging.error(f"Hugging Face image generation failed: {e}")
        msg = str(e)
        if "402" in msg or "Payment Required" in msg:
            return {"error": "Hugging Face account has depleted credits. Upgrade or buy more."}, 402
        return {"error": f"Image generation failed: {msg}"}, 500

    try:
        return encode_image(add_dialogue(image, panel.get("dialogue", ""))), 200
    except Exception as e:
        logging.error(f"Adding dialogue failed: {e}")
        return {"error": f"Dialogue overlay failed: {str(e)}"}, 500

# ================== PIPELINE ==================
def generate_comic_pipeline(story, style):
    panels = generate_story(story)
    images = []

    for idx, panel in enumerate(panels, start=1):
        img_result, status = generate_image(panel, style)
        if status != 200:
            return {"status": "error", "message": img_result.get("error", "Unknown error")}, status
        images.append(img_result)

    return {"status": "success", "panels": panels, "images": images}, 200

# ================== ROUTES ==================
@app.route("/output", methods=["POST"])
def output():
    data = request.json
    story = data.get("text", "").strip()
    style = data.get("image_style", "cartoonish")

    if not story:
        return jsonify({"status": "error", "message": "No story provided"}), 400

    try:
        result, status = generate_comic_pipeline(story, style)
        return jsonify(result), status
    except Exception as e:
        logging.exception("Unexpected server error")
        return jsonify({"status": "error", "message": "Unexpected server error"}), 500

@app.route("/")
def health():
    return {
        "status": "running",
        "hf_loaded": hf_client is not None,
        "gemini_loaded": gemini_client is not None,
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
