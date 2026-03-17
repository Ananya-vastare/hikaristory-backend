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

# ================== UTILITIES ==================
def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def wrap_text(draw, text, font, max_width):
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
    return lines

def add_dialogue(image: Image.Image, dialogue: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    padding = 10
    max_width = image.width - 60

    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        font = ImageFont.load_default()

    lines = wrap_text(draw, dialogue, font, max_width)
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    text_height = len(lines) * (line_height + 3) + padding*2

    bubble_x0, bubble_y0 = 30, 30
    bubble_x1 = bubble_x0 + max([draw.textbbox((0,0), line, font=font)[2] for line in lines]) + padding*2
    bubble_y1 = bubble_y0 + text_height

    # Draw bubble
    draw.rounded_rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1],
                           radius=15, fill="white", outline="black", width=3)

    # Bubble tail
    tail_width, tail_height = 30, 20
    tail_x0 = bubble_x0 + bubble_x1 // 4
    tail_y0 = bubble_y1
    draw.polygon([(tail_x0, tail_y0),
                  (tail_x0 + tail_width, tail_y0),
                  (tail_x0 + tail_width//2, tail_y0 + tail_height)],
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

# ================== IMAGE GENERATION ==================
def generate_image(panel, style):
    width, height = 600, 400  # safe image size

    if not hf_client:
        # fallback simple image
        image = Image.new("RGB", (width, height), color="skyblue")
        image = add_dialogue(image, panel.get("dialogue", ""))
        return encode_image(image), None

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
            width=width,
            height=height,
        )
    except Exception as e:
        logging.warning(f"HF image generation failed: {str(e)}. Using placeholder.")
        image = Image.new("RGB", (width, height), color="skyblue")

    # Add dialogue safely
    try:
        image = add_dialogue(image, panel.get("dialogue", ""))
    except Exception as e:
        logging.warning(f"Failed to add dialogue: {str(e)}")

    return encode_image(image), None

# ================== PIPELINE ==================
def generate_comic_pipeline(story, style):
    panels, err = generate_story(story)
    if err:
        return None, err

    images = []
    for panel in panels:
        logging.info(f"Generating panel: {panel.get('scene','')[:40]}...")
        try:
            img, _ = generate_image(panel, style)
        except Exception as e:
            logging.warning(f"Panel generation failed, using placeholder: {str(e)}")
            img = encode_image(Image.new("RGB", (600, 400), color="skyblue"))
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
