"""
generate_assets.py — Generate placeholder VN assets.

Creates:
  1. Placeholder background images for all bg keys
  2. Placeholder character sprites for all curated characters × expressions
  3. Fallback sprite set for the curated cast

Run from the ars-fabula-vn directory:
    python tools/generate_assets.py

Requires: Pillow (PIL)
"""
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers import PRESET_BACKGROUNDS, BG_DIR
from cast_pipeline import PRESET_CHARACTERS, EXPRESSION_TAGS, STANDARD_EXPRESSIONS
from cast_pipeline import CURATED_DIR, GENERATED_DIR
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════════════════════
# Backgrounds
# ═══════════════════════════════════════════════════════════════════════

def _draw_gradient(draw, width, height, top_color, bottom_color):
    """Draw a vertical gradient."""
    for y in range(height):
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * y / height)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * y / height)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * y / height)
        draw.line([(0, y), (width, y)], fill=(r, g, b))


def _add_noise(draw, width, height, intensity=10):
    """Add subtle noise for texture."""
    import random
    for _ in range(width * height // 100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r = random.randint(-intensity, intensity)
        g = random.randint(-intensity, intensity)
        b = random.randint(-intensity, intensity)
        try:
            px = draw.im.getpixel((x, y))
            draw.point((x, y), fill=(
                min(255, max(0, px[0] + r)),
                min(255, max(0, px[1] + g)),
                min(255, max(0, px[2] + b)),
            ))
        except Exception:
            pass


def generate_backgrounds():
    """Generate all placeholder background images."""
    WIDTH, HEIGHT = 1024, 768
    os.makedirs(BG_DIR, exist_ok=True)

    bg_colors = {
        "classroom":     ((180, 175, 140), (140, 135, 100)),  # warm beige
        "hallway":       ((160, 165, 150), (120, 125, 110)),  # cool beige
        "rooftop":       ((200, 180, 120), (100, 120, 160)),  # golden → sky
        "courtyard":     ((150, 190, 120), (100, 140, 80)),   # green
        "cafe":          ((190, 160, 120), (140, 110, 80)),   # warm brown
        "park":          ((140, 180, 110), (100, 140, 80)),   # park green
        "library":       ((170, 160, 130), (120, 110, 90)),   # book brown
        "home_room":     ((180, 150, 140), (130, 100, 90)),   # warm pink
        "gate":          ((180, 170, 120), (140, 110, 160)),  # golden → purple
        "sunset_hill":   ((220, 150, 100), (100, 80, 140)),   # sunset
    }

    for key, (top, bottom) in bg_colors.items():
        path = os.path.join(BG_DIR, f"bg_{key}.png")
        if os.path.exists(path):
            print(f"  [bg] {key} — already exists")
            continue

        img = Image.new("RGB", (WIDTH, HEIGHT), top)
        draw = ImageDraw.Draw(img)
        _draw_gradient(draw, WIDTH, HEIGHT, top, bottom)
        _add_noise(draw, WIDTH, HEIGHT, 8)

        # Add label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((WIDTH // 2, HEIGHT - 30),
                  f"• {key.replace('_', ' ').title()} •",
                  fill=(255, 255, 255, 180), font=font, anchor="ms")

        img.save(path)
        print(f"  [bg] {key} — generated ({path})")


# ═══════════════════════════════════════════════════════════════════════
# Character Sprites
# ═══════════════════════════════════════════════════════════════════════

def _draw_circle(draw, cx, cy, r, fill):
    """Draw a filled circle."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def _draw_eyes(draw, x, y, expression):
    """Draw anime-style eyes."""
    eye_spacing = 40
    eye_y = y
    left_x = x - eye_spacing // 2
    right_x = x + eye_spacing // 2

    if expression in ("surprised",):
        # Wide round eyes
        for ex in (left_x, right_x):
            _draw_circle(draw, ex, eye_y, 15, (60, 60, 80))
            _draw_circle(draw, ex, eye_y, 10, (40, 40, 60))
            draw.point((ex + 2, eye_y - 2), fill=(255, 255, 255))
    elif expression in ("sad",):
        # Downward half-closed eyes
        for ex in (left_x, right_x):
            draw.arc([ex - 15, eye_y - 10, ex + 15, eye_y + 10],
                     0, 180, fill=(60, 60, 80), width=3)
    elif expression in ("angry",):
        # Angled angry eyes
        for ex in (left_x, right_x):
            draw.arc([ex - 15, eye_y - 10, ex + 15, eye_y + 10],
                     180, 360, fill=(80, 50, 50), width=3)
    elif expression in ("embarrassed",):
        # Spiral eyes (@_@)
        for ex in (left_x, right_x):
            draw.arc([ex - 12, eye_y - 10, ex + 12, eye_y + 10],
                     0, 360, fill=(80, 60, 60), width=2)
            draw.line([(ex - 6, eye_y - 6), (ex + 6, eye_y + 6)], fill=(80, 60, 60), width=1)
            draw.line([(ex + 6, eye_y - 6), (ex - 6, eye_y + 6)], fill=(80, 60, 60), width=1)
    else:
        # Normal eyes
        for ex in (left_x, right_x):
            draw.arc([ex - 15, eye_y - 8, ex + 15, eye_y + 8],
                     0, 180, fill=(60, 60, 80), width=3)
            _draw_circle(draw, ex + 3, eye_y - 1, 4, (40, 40, 60))
            draw.point((ex + 4, eye_y - 3), fill=(255, 255, 255))


def _draw_mouth(draw, x, y, expression):
    """Draw anime-style mouth."""
    if expression == "smile":
        draw.arc([x - 12, y - 5, x + 12, y + 10], 0, 180, fill=(80, 50, 50), width=2)
    elif expression == "laugh":
        draw.chord([x - 15, y - 8, x + 15, y + 12], 0, 180, fill=(60, 40, 40))
    elif expression == "sad":
        draw.arc([x - 10, y - 5, x + 10, y + 8], 180, 360, fill=(60, 50, 50), width=2)
    elif expression == "angry":
        draw.arc([x - 10, y - 8, x + 10, y + 5], 0, 180, fill=(80, 40, 40), width=2)
        # Teeth gritting
        draw.line([(x - 6, y - 1), (x + 6, y - 1)], fill=(200, 200, 200), width=2)
    elif expression == "surprised":
        _draw_circle(draw, x, y, 6, (40, 30, 30))
    elif expression == "embarrassed":
        draw.arc([x - 8, y - 3, x + 8, y + 6], 0, 180, fill=(80, 50, 50), width=2)
    else:
        # Neutral — small line
        draw.line([(x - 6, y), (x + 6, y)], fill=(60, 50, 50), width=2)


def generate_sprites():
    """Generate placeholder character sprites for all characters × expressions."""
    sprite_w, sprite_h = 512, 768  # Match CastMember default

    char_colors = {
        "yuki": {
            "skin": (240, 220, 210),
            "hair": (200, 210, 230),  # silver
            "eyes": (180, 80, 80),    # red
            "uniform": ((200, 210, 230), (100, 120, 180)),
        },
        "haruki": {
            "skin": (230, 210, 200),
            "hair": (40, 40, 50),     # black
            "eyes": (120, 90, 60),    # brown
            "uniform": ((200, 210, 230), (180, 180, 200)),
        },
        "sakura": {
            "skin": (245, 220, 215),
            "hair": (220, 160, 180),  # pink
            "eyes": (100, 180, 100),  # green
            "uniform": ((200, 210, 230), (200, 150, 170)),
        },
        "ryo": {
            "skin": (235, 215, 205),
            "hair": (60, 60, 100),    # blue
            "eyes": (80, 80, 120),    # blue-gray
            "uniform": ((200, 210, 230), (100, 100, 140)),
        },
    }

    for char_key, preset in PRESET_CHARACTERS.items():
        colors = char_colors.get(char_key, char_colors["yuki"])
        char_dir = os.path.join(CURATED_DIR, char_key)
        os.makedirs(char_dir, exist_ok=True)

        for expr in STANDARD_EXPRESSIONS:
            path = os.path.join(char_dir, f"{char_key}_{expr}.png")
            if os.path.exists(path):
                print(f"  [sprite] {char_key}/{expr} — already exists")
                continue

            # Create character sprite
            img = Image.new("RGBA", (sprite_w, sprite_h), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # --- Body / uniform ---
            uni_top, uni_bottom = colors["uniform"]

            # Torso
            draw.rectangle([160, 280, 352, 550], fill=(*uni_top, 255), outline=(*uni_bottom, 255), width=2)
            # Collar
            draw.polygon([(200, 280), (256, 320), (312, 280)], fill=(*uni_top, 255), outline=(*uni_bottom, 255))

            # --- Head ---
            head_cx, head_cy = 256, 180
            face_color = (*colors["skin"], 255)

            # Hair
            hair_color = (*colors["hair"], 255)
            draw.ellipse([head_cx - 70, head_cy - 80, head_cx + 70, head_cy + 60], fill=hair_color)

            # Face oval
            draw.ellipse([head_cx - 55, head_cy - 60, head_cx + 55, head_cy + 70], fill=face_color)

            # Hair bangs
            draw.polygon([
                (head_cx - 60, head_cy - 40),
                (head_cx - 40, head_cy + 10),
                (head_cx - 20, head_cy - 20),
                (head_cx, head_cy + 5),
                (head_cx + 20, head_cy - 20),
                (head_cx + 40, head_cy + 10),
                (head_cx + 60, head_cy - 40),
            ], fill=hair_color)

            # Hair side strands
            draw.ellipse([head_cx - 80, head_cy - 20, head_cx - 50, head_cy + 40], fill=hair_color)
            draw.ellipse([head_cx + 50, head_cy - 20, head_cx + 80, head_cy + 40], fill=hair_color)

            # --- Eyes ---
            _draw_eyes(draw, head_cx, head_cy + 5, expr)

            # --- Eyebrows ---
            brow_y = head_cy - 18
            if expr in ("sad", "embarrassed"):
                # Raised inner, slanted up
                draw.line([(head_cx - 35, brow_y + 5), (head_cx - 10, brow_y - 2)], fill=(60, 50, 40), width=3)
                draw.line([(head_cx + 10, brow_y - 2), (head_cx + 35, brow_y + 5)], fill=(60, 50, 40), width=3)
            elif expr in ("angry",):
                # Angled down
                draw.line([(head_cx - 35, brow_y - 3), (head_cx - 10, brow_y + 5)], fill=(60, 40, 40), width=3)
                draw.line([(head_cx + 10, brow_y + 5), (head_cx + 35, brow_y - 3)], fill=(60, 40, 40), width=3)
            elif expr in ("surprised",):
                # Arched up
                draw.line([(head_cx - 35, brow_y - 5), (head_cx - 10, brow_y - 10)], fill=(60, 50, 40), width=3)
                draw.line([(head_cx + 10, brow_y - 10), (head_cx + 35, brow_y - 5)], fill=(60, 50, 40), width=3)
            else:
                # Normal
                draw.line([(head_cx - 35, brow_y), (head_cx - 10, brow_y)], fill=(60, 50, 40), width=3)
                draw.line([(head_cx + 10, brow_y), (head_cx + 35, brow_y)], fill=(60, 50, 40), width=3)

            # --- Mouth ---
            _draw_mouth(draw, head_cx, head_cy + 30, expr)

            # --- Blush ---
            if expr in ("embarrassed", "smile"):
                blush_color = (220, 120, 120, 80)
                draw.ellipse([head_cx - 35, head_cy + 20, head_cx - 10, head_cy + 35], fill=blush_color)
                draw.ellipse([head_cx + 10, head_cy + 20, head_cx + 35, head_cy + 35], fill=blush_color)

            # --- Tear for sad ---
            if expr in ("sad",):
                tear_color = (150, 180, 220, 120)
                draw.ellipse([head_cx - 45, head_cy + 15, head_cx - 30, head_cy + 35], fill=tear_color)
                draw.ellipse([head_cx + 30, head_cy + 15, head_cx + 45, head_cy + 35], fill=tear_color)

            # --- Neck ---
            draw.rectangle([head_cx - 15, head_cy + 65, head_cx + 15, head_cy + 85], fill=face_color)

            # --- Shoulders ---
            draw.arc([120, 260, 392, 400], 180, 0, fill=face_color, width=4)

            # --- Arms ---
            arm_color = (*uni_top, 255)
            draw.rectangle([130, 320, 155, 480], fill=arm_color, outline=(*uni_bottom, 255))
            draw.rectangle([357, 320, 382, 480], fill=arm_color, outline=(*uni_bottom, 255))

            # Add character name and expression label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

            draw.text((sprite_w // 2, sprite_h - 20),
                      f"{char_key} — {expr}",
                      fill=(150, 150, 150, 180), font=font, anchor="ms")

            img.save(path)
            print(f"  [sprite] {char_key}/{expr} — generated")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎨 Generating VN placeholder assets...\n")

    print("--- Backgrounds ---")
    generate_backgrounds()

    print("\n--- Character Sprites ---")
    generate_sprites()

    print("\n✅ All assets generated!")
    print(f"   Backgrounds: {BG_DIR}")
    print(f"   Sprites: {CURATED_DIR}")
