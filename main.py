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
        if bbox[2] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def add_description_and_dialogue(image: Image.Image, description: str, dialogue: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # --- DESCRIPTION BOX ---
    desc_font_size = max(30, img_w // 25)
    try:
        desc_font = ImageFont.truetype("arial.ttf", desc_font_size)
    except:
        desc_font = ImageFont.load_default()

    padding = desc_font_size // 2
    max_width = img_w - 100

    desc_lines = wrap_text(draw, description, desc_font, max_width)
    line_height = desc_font.getbbox("A")[3] - desc_font.getbbox("A")[1]
    desc_height = len(desc_lines) * line_height + padding * 2
    desc_box = [50, 20, img_w - 50, 20 + desc_height]
    draw.rectangle(desc_box, fill="lightyellow", outline="black", width=3)

    y = desc_box[1] + padding
    for line in desc_lines:
        draw.text((desc_box[0] + padding, y), line, fill="black", font=desc_font)
        y += line_height + 5

    # --- DIALOGUE BUBBLE ---
    dialogue_font_size = max(40, img_w // 20)
    try:
        font = ImageFont.truetype("arial.ttf", dialogue_font_size)
    except:
        font = ImageFont.load_default()

    bubble_padding = dialogue_font_size // 2
    lines = wrap_text(draw, dialogue, font, max_width - 20)
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    text_height = len(lines) * line_height + bubble_padding * 2
    text_width = max(draw.textbbox((0,0), line, font=font)[2] for line in lines) + bubble_padding * 2

    bubble_x0 = 50
    bubble_y0 = desc_box[3] + 40
    bubble_x1 = bubble_x0 + text_width
    bubble_y1 = bubble_y0 + text_height

    # Draw ellipse bubble
    draw.ellipse([bubble_x0, bubble_y0, bubble_x1, bubble_y1], fill="white", outline="black", width=4)

    # Tail
    tail_w = text_width // 4
    tail_h = tail_w // 2
    tail_x0 = bubble_x0 + tail_w
    tail_y0 = bubble_y1
    draw.polygon([
        (tail_x0, tail_y0),
        (tail_x0 + tail_w, tail_y0),
        (tail_x0 + tail_w // 2, tail_y0 + tail_h)
    ], fill="white", outline="black")

    # Draw dialogue text
    y = bubble_y0 + bubble_padding
    for line in lines:
        draw.text((bubble_x0 + bubble_padding, y), line, fill="black", font=font)
        y += line_height + 5

    return image

# ================== STORY GENERATION ==================
def generate_story(story_text):
    if not gemini_client:
        return None, None, "Gemini API key missing"

    prompt = f"""
Create a 4-panel comic about this story: {story_text}

Rules:
- Same character in all panels
- Describe the character once
- Each panel has a cinematic scene and a 1-sentence dialogue
- Return JSON ONLY

Format:
{{
 "character": "detailed description of main character",
 "panels":[
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}},
   {{"scene":"...", "dialogue":"..."}}
 ]
}}
"""
    try:
        res = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        text = res.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        data = json.loads(text)
        return data["character"], data["panels"], None
    except Exception as e:
        logging.error(f"Gemini error: {str(e)}")
        return None, None, "Story generation failed"

# ================== IMAGE GENERATION ==================
def generate_image(panel, character_desc, style):
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
Character: {character_desc}
Scene: {panel['scene']}
"""
    try:
        image = hf_client.text_to_image(
            prompt=prompt,
            negative_prompt="blurry, distorted, bad anatomy",
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=800,
            height=800
        )
    except Exception as e:
        logging.warning(f"Primary model failed, retrying: {str(e)}")
        try:
            image = hf_client.text_to_image(
                prompt=prompt,
                negative_prompt="blurry, distorted",
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=1024,
                height=1024
            )
        except Exception as e2:
            logging.error(f"HuggingFace error: {str(e2)}")
            return None, "Image generation failed"

    # Add proper description + bubble
    try:
        image = add_description_and_dialogue(
            image, panel.get("scene", ""), panel.get("dialogue", "")
        )
    except Exception as e:
        logging.warning(f"Failed to add dialogue: {str(e)}")

    return encode_image(image), None

# ================== PIPELINE ==================
def generate_comic_pipeline(story, style):
    character_desc, panels, err = generate_story(story)
    if err:
        return None, err

    images = []
    for panel in panels:
        logging.info(f"Generating image: {panel['scene'][:40]}...")
        img, err = generate_image(panel, character_desc, style)
        if err:
            return None, err
        images.append(img)

    return {"character": character_desc, "panels": panels, "images": images}, None

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
        "gemini_loaded": bool(GEMINI_API_KEY)
    }

# ================== RUN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
