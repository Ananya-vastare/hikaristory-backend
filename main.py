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

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_API_KEY or not GEMINI_API_KEY:
    raise Exception("Missing API keys in .env")

app = Flask(__name__)
CORS(app)

hf_client = InferenceClient(api_key=HF_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Premium animated style prompt
FIXED_ART_PROMPT = (
    "ultra-detailed, premium animated movie still, cinematic lighting, "
    "dynamic poses, dramatic shadows, depth of field, high resolution, vibrant colors, "
    "soft lighting, realistic textures, anime-inspired but photorealistic, studio-quality render"
)


def add_dialogue_bubble(image: Image.Image, dialogue: str) -> Image.Image:
    """Optional subtle dialogue bubble."""
    if not dialogue:
        return image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    words = dialogue.split()
    lines, current = [], ""
    max_width = 400
    for word in words:
        test = f"{current} {word}".strip() if current else word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] < max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    bubble_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 25
    bubble_height = len(lines) * 28 + 30
    bubble_x, bubble_y = 20, 20

    # Semi-transparent, subtle, cinematic dialogue bubble
    draw.rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
        fill=(255, 255, 255, 180),
        outline=(50, 50, 50),
        width=3,
    )

    y = bubble_y + 5
    for line in lines:
        draw.text((bubble_x + 10, y), line, fill="black", font=font)
        y += 28

    return image


def generate_story_panels(story: str):
    """Generate 4-panel cinematic animated story with scene + dialogue."""
    prompt = f"""
You are a professional animated movie concept artist and writer.
Create a coherent 4-panel cinematic animated story.
Rules:
- All panels follow same main characters.
- Each panel advances the story logically.
- Each panel must include:
    1) Scene: detailed animated movie still description
    2) Dialogue: 1-2 sentences characters speak
Return JSON only:
{{
 "panels":[
  {{"scene":"description","dialogue":"text"}},
  {{"scene":"description","dialogue":"text"}},
  {{"scene":"description","dialogue":"text"}},
  {{"scene":"description","dialogue":"text"}}
 ]
}}
Story Idea:
{story}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        data = json.loads(text)
        return data["panels"][:4], None
    except Exception as e:
        print("GEMINI ERROR:", e)
        return None, "GEMINI_GENERATION_ERROR"


def generate_panel_images(panels: list):
    """Generate high-resolution animated-style images with optional dialogue."""
    images = []
    try:
        for panel in panels:
            prompt = f"{FIXED_ART_PROMPT}, {panel.get('scene','')}"
            image = hf_client.text_to_image(
                prompt,
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=1024,  # high-res for premium quality
                height=1024,
                guidance_scale=15.0,
                num_inference_steps=150,
            )

            # Optional dialogue bubble
            image = add_dialogue_bubble(image, panel.get("dialogue", ""))

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            images.append(base64.b64encode(buffer.getvalue()).decode())
        return images, None
    except Exception as e:
        print("HF ERROR:", e)
        return None, "HF_GENERATION_ERROR"


@app.route("/", method=["GET"])
def home():
    return jsonify({{"message": "Everything is happening"}})


@app.route("/output", methods=["POST"])
def generate_comic():
    data = request.json
    story = data.get("text", "")
    if not story:
        return jsonify({"status": "error", "message": "No storyline provided"}), 400

    panels, gem_error = generate_story_panels(story)
    if gem_error:
        return jsonify({"status": "error", "source": "Gemini", "error": gem_error}), 500

    images, hf_error = generate_panel_images(panels)
    if hf_error:
        return (
            jsonify({"status": "error", "source": "HuggingFace", "error": hf_error}),
            500,
        )

    result = [
        {"scene": p["scene"], "dialogue": p["dialogue"], "image": img}
        for p, img in zip(panels, images)
    ]
    return jsonify({"status": "success", "panels": result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
