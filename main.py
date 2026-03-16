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
import concurrent.futures
import traceback
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logging.info(f"HuggingFace key loaded: {bool(HF_API_KEY)}")
logging.info(f"Gemini key loaded: {bool(GEMINI_API_KEY)}")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def add_dialogue(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    bubble_x, bubble_y = 40, 40
    bubble_width, bubble_height = 500, 210

    draw.rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
        fill="white",
        outline="black",
        width=4,
    )

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] < bubble_width - 40:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    y = bubble_y + 20
    for line in lines:
        draw.text((bubble_x + 15, y), line, fill="black", font=font)
        y += 32

    return image


def generate_story_panels(story_text):
    if not gemini_client:
        return None, {
            "status": "error",
            "source": "Gemini",
            "error": "GEMINI_CLIENT_NOT_INITIALIZED",
            "message": "Gemini client is not initialized.",
        }

    prompt = f"""
You are a professional comic writer.
Create a coherent 4-panel comic story.
Rules:
- Same characters across panels
- Story progresses logically
- Dialogue max 2 sentences
- Return ONLY JSON

Format:
{{
 "panels":[
  {{"scene":"visual description","dialogue":"character dialogue"}},
  {{"scene":"visual description","dialogue":"character dialogue"}},
  {{"scene":"visual description","dialogue":"character dialogue"}},
  {{"scene":"visual description","dialogue":"character dialogue"}}
 ]
}}

Story Idea:
{story_text}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        data = json.loads(text)
        panels = data.get("panels", [])[:4]
        return panels, None
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(
            f"Gemini generation failed.\nError: {e}\nTraceback:\n{tb}\nInput: {story_text}"
        )
        return None, {
            "status": "error",
            "source": "Gemini",
            "error": type(e).__name__,
            "message": str(e),
            "traceback": tb,
            "input_preview": story_text[:100],
        }


def generate_single_image(panel, style_map, style):
    style_prompt = style_map.get(style, style_map["cartoonish"])
    prompt = f"{style_prompt}, {panel['scene']}"
    try:
        image = hf_client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=512,
            height=512,
        )
        image = add_dialogue(image, panel["dialogue"])
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64, None
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(
            f"HuggingFace generation failed.\nError: {e}\nTraceback:\n{tb}\nPanel: {panel}"
        )
        return None, {
            "status": "error",
            "source": "HuggingFace",
            "error": type(e).__name__,
            "message": str(e),
            "traceback": tb,
            "panel_preview": panel,
        }


def generate_images(panels, style="cartoonish"):
    if not hf_client:
        return None, {
            "status": "error",
            "source": "HuggingFace",
            "error": "HF_CLIENT_NOT_INITIALIZED",
            "message": "HuggingFace client is not initialized.",
        }

    style_map = {
        "cartoonish": "comic book illustration, bold outlines, vibrant colors",
        "soft": "soft watercolor illustration, gentle pastel colors",
        "dramatic": "dramatic cinematic lighting, high contrast",
        "pixel_art": "pixel art retro game style",
        "manga": "manga style black and white comic",
    }

    images = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate_single_image, panel, style_map, style)
                for panel in panels
            ]
            for future in concurrent.futures.as_completed(futures):
                img, img_error = future.result()
                if img_error:
                    # Return the first image generation error
                    return None, img_error
                images.append(img)
        return images, None
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(
            f"HuggingFace generation failed.\nError: {e}\nTraceback:\n{tb}\nPanels: {panels}"
        )
        return None, {
            "status": "error",
            "source": "HuggingFace",
            "error": type(e).__name__,
            "message": str(e),
            "traceback": tb,
            "panels_preview": panels[:2],
        }


@app.route("/output", methods=["POST"])
def generate_comic():
    try:
        data = request.json
        story = data.get("text", "")
        style = data.get("image_style", "cartoonish")

        if not story:
            return jsonify({"status": "error", "message": "No storyline provided"}), 400

        panels, gemini_error = generate_story_panels(story)
        if gemini_error:
            return jsonify(gemini_error), 500

        images, hf_error = generate_images(panels, style)
        if hf_error:
            return jsonify(hf_error), 500

        return jsonify({"status": "success", "panels": panels, "images": images})

    except Exception as e:
        tb = traceback.format_exc()
        logging.exception("SERVER ERROR")
        return (
            jsonify(
                {
                    "status": "error",
                    "source": "Server",
                    "error": type(e).__name__,
                    "message": str(e),
                    "traceback": tb,
                }
            ),
            500,
        )


@app.route("/", methods=["GET"])
def health():
    return {
        "message": "Comic backend running",
        "HF_API_KEY_loaded": bool(HF_API_KEY),
        "GEMINI_API_KEY_loaded": bool(GEMINI_API_KEY),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
