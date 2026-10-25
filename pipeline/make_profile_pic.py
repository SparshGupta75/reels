#!/usr/bin/env python3
"""Draws the profile picture for @mindblownreels. Run once, then upload reels/profile-pic.png."""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(os.path.dirname(ROOT), "explainer", "Anton-Regular.ttf")
AMBER = (255, 214, 10)
INK = (12, 12, 12)
S = 1080


def centred(draw, text, font, y, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((S - (box[2] - box[0])) / 2 - box[0], y - box[1]), text, font=font, fill=fill)


def main():
    img = Image.new("RGB", (S, S), AMBER)
    draw = ImageDraw.Draw(img)
    big = ImageFont.truetype(FONT, 430)
    small = ImageFont.truetype(FONT, 64)
    centred(draw, "WHY?", big, 250, INK)
    draw.rectangle([180, 760, S - 180, 776], fill=INK)
    centred(draw, "EVERYDAY THINGS", small, 830, INK)
    img.save(os.path.join(ROOT, "profile-pic.png"))
    img.resize((160, 160), Image.LANCZOS).save(os.path.join(ROOT, "profile-pic-small.png"))
    print("wrote profile-pic.png and profile-pic-small.png")


if __name__ == "__main__":
    main()
