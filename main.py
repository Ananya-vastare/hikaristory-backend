from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from google import genai
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from dotenv import load_dotenv
import base64
import json
import os
import logging
import traceback
import concurrent.futures

load_dotenv()

logging.basicConfig(level=logging.INFO)

HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def draw_speech_bubble(image: Image.Image, text: str):

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    max_width = 300
    padding = 14
    line_height = 26

    # wrap text
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

    # calculate bubble size
    text_width = max(draw.textbbox((0, 0), l, font=font)[2] for l in lines)
    bubble_width = text_width + padding * 2
    bubble_height = len(lines) * line_height + padding * 2

    x = 30
    y = 30

    # bubble
    draw.rounded_rectangle(
        (x, y, x + bubble_width, y + bubble_height),
        radius=18,
        fill="white",
        outline="black",
        width=3
    )

    # speech tail
    tail = [
        (x + 70, y + bubble_height),
        (x + 95, y + bubble_height),
        (x + 80, y + bubble_height + 30)
    ]
    draw.polygon(tail, fill="white", outline="black")

    # draw text
    ty = y + padding
    for line in lines:
        draw.text((x + padding, ty), line, fill="black", font=font)
        ty += line_height

    return image


def generate_story_panels(story_text):

    if not gemini_client:
        raise RuntimeError("Gemini client not initialized")

    prompt = f"""
Create a 4 panel comic story.

Rules:
- Same characters in all panels
- Story must progress logically
- Dialogue max 1 sentence
- Return ONLY JSON

Format:
{{
"panels":[
{{"scene":"visual description","dialogue":"speech"}},
{{"scene":"visual description","dialogue":"speech"}},
{{"scene":"visual description","dialogue":"speech"}},
{{"scene":"visual description","dialogue":"speech"}}
]
}}

Story idea:
{story_text}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    data = json.loads(text)

    return data["panels"][:4]


STYLE_MAP = {
    "cartoonish": "comic book illustration, bold outlines, colorful",
    "soft": "soft watercolor illustration",
    "dramatic": "cinematic lighting dramatic illustration",
    "manga": "black and white manga panel",
    "pixel": "retro pixel art style"
}


def generate_panel_image(panel, style):

    prompt = f"{STYLE_MAP.get(style,'comic illustration')}, {panel['scene']}"

    image = hf_client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=512,
        height=512
    )

    image = draw_speech_bubble(image, panel["dialogue"])

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def generate_images(panels, style):

    images = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(generate_panel_image, panel, style)
            for panel in panels
        ]

        for f in futures:
            images.append(f.result())

    return images

@app.route("/output", methods=["POST"])
def generate_comic():

    try:

        data = request.json
        story = data.get("text", "")
        style = data.get("image_style", "cartoonish")

        if not story:
            return jsonify({"error": "Story text missing"}), 400

        panels = generate_story_panels(story)

        images = generate_images(panels, style)

        return jsonify({
            "status": "success",
            "panels": panels,
            "images": images
        })

    except Exception as e:

        logging.error(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/", methods=["GET"])
def health():

    return {
        "service": "Comic Generator Backend",
        "huggingface_loaded": bool(HF_API_KEY),
        "gemini_loaded": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
