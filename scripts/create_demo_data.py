from PIL import Image, ImageDraw, ImageFont
import os
import json
import numpy as np

samples = [
    {"text": "本花あ", "filename": "demo_01.jpg"},
    {"text": "学道力", "filename": "demo_02.jpg"},
    {"text": "心空夢", "filename": "demo_03.jpg"},
]

IMG_SIZE = (640, 160)
SAVE_ROOT = os.path.join("data", "demo")

# Pfad zu einer TTF-Schriftart, die japanische Zeichen unterstützt
# z.B. auf Windows: C:/Windows/Fonts/msgothic.ttc
# auf Linux: /usr/share/fonts/truetype/fonts-japanese-gothic.ttf
# auf Mac: /System/Library/Fonts/ヒラギノ角ゴシック W3.ttc
FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"
FONT_SIZE = 80

def ensure_dirs():
    os.makedirs(os.path.join(SAVE_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "annotations"), exist_ok=True)

def create_demo_image(text, filename):
    img = Image.new("RGB", IMG_SIZE, color=0xFFFFFF)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    boxes, labels = [], []
    x, y = 30, 40
    spacing = 150

    for ch in text:
        draw.text((x, y), ch, font=font, fill="black")
        boxes.append([x, y, x +5, y +5])
        labels.append(ch)
        x += spacing

    # Bild speichern
    img_path = os.path.join(SAVE_ROOT, "images", filename)
    img.save(img_path)

    # JSON-Annotation
    ann = {"boxes": boxes, "labels": labels}
    ann_path = os.path.join(SAVE_ROOT, "annotations", filename.replace(".jpg", ".json"))
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

def main():
    ensure_dirs()
    for sample in samples:
        create_demo_image(sample["text"], sample["filename"])
    print("✅ Demo-Dataset erstellt unter:", SAVE_ROOT)

if __name__ == "__main__":
    main()
