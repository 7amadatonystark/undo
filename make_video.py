"""
make_video.py — Step V1: turn the scripted demo (video_demo.py) into a video file.

It runs the scripted demo (no API calls, the same every time), takes a picture of the screen at
every pause, and writes the pictures into media/undo_demo.mp4 (1920x1080), with a title card at
the start and a results card at the end. The pauses are a little longer than in the terminal,
so a narrator has time to talk.

    python make_video.py              # media/undo_demo.mp4 (about 68 s)
    python make_video.py --speed 2    # media/undo_demo_x2.mp4: twice as fast (about 36 s)
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

import video_demo

OUT = Path(__file__).parent / "media" / "undo_demo.mp4"
WIDTH, HEIGHT, FPS = 1920, 1080, 10
STRETCH = 1.4                       # each pause lasts 1.4x longer than in the terminal
COLUMNS = 86                        # the demo's text width, in characters
BACKGROUND = (13, 17, 23)
TEXT = (230, 237, 243)
FONTS = Path("C:/Windows/Fonts")
MONO = ImageFont.truetype(str(FONTS / "consola.ttf"), 27)
MONO_BOLD = ImageFont.truetype(str(FONTS / "consolab.ttf"), 27)
TITLE = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 110)
SUBTITLE = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 44)
CARD = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 50)
CARD_BOLD = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 50)
LINE_HEIGHT = 32                    # = the font's box-character height, so borders join up
NAMED = {"green": (63, 185, 80), "yellow": (210, 153, 34), "red": (248, 81, 73),
         "cyan": (57, 197, 207), "white": TEXT}

TITLE_CARD = ("UNDO", "A safety layer for AI agents",
              "Scripted demonstration of the UNDO runtime.")
RESULTS_CARD = [("On 12 tasks it had never seen,", False),
                ("damage left by the AI agent fell from", False),
                ("39% without UNDO to 9% with it.", True),
                ("", False),
                ("A person was asked 83% less often", False),
                ("than if every action needed approval.", False)]


def color_of(style):
    """The RGB color of a rich style (the demo uses a few named colors)."""
    if style and style.color:
        name = style.color.name
        if name in NAMED:
            return NAMED[name]
        triplet = style.color.get_truecolor()
        return (triplet.red, triplet.green, triplet.blue)
    if style and style.dim:
        return (139, 148, 158)
    return TEXT


def lines_of(segments):
    """Rich's recorded segments -> lines of (text, color, bold) pieces."""
    lines, current = [], []
    for segment in segments:
        if segment.control:
            continue
        parts = segment.text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append(current)
                current = []
            if part:
                current.append((part, color_of(segment.style), bool(segment.style and segment.style.bold)))
    lines.append(current)
    return lines


def screen(segments):
    """One frame: the demo's text so far, the newest lines at the bottom if it is too long."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    char = MONO.getlength("M")
    left = (WIDTH - COLUMNS * char) / 2
    lines = lines_of(segments)
    lines = lines[-(HEIGHT // LINE_HEIGHT - 2):]
    for row, pieces in enumerate(lines):
        x, y = left, 40 + row * LINE_HEIGHT
        for text, color, bold in pieces:
            draw.text((x, y), text, font=MONO_BOLD if bold else MONO, fill=color)
            x += len(text) * char
    return np.asarray(image)


def card(lines, fonts):
    """A centered text card."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    heights = [fonts[i].size * 1.45 for i in range(len(lines))]
    y = (HEIGHT - sum(heights)) / 2
    for i, (text, color) in enumerate(lines):
        width = draw.textlength(text, font=fonts[i])
        draw.text(((WIDTH - width) / 2, y), text, font=fonts[i], fill=color)
        y += heights[i]
    return np.asarray(image)


def record_demo():
    """Run the scripted demo, keeping (frame, seconds) at every pause."""
    console = Console(record=True, width=COLUMNS, force_terminal=True, color_system="truecolor",
                      file=open("nul" if Path("C:/").exists() else "/dev/null", "w"))
    shots = []

    def pause(seconds=video_demo.PAUSE):
        shots.append((list(console._record_buffer), seconds * STRETCH))
    video_demo.console = console
    video_demo.pause = pause
    video_demo.main()
    shots.append((list(console._record_buffer), 5.0))        # hold the last screen
    return shots


def main():
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    out = OUT if speed == 1 else OUT.with_name(f"undo_demo_x{speed:g}.mp4")
    shots = record_demo()
    OUT.parent.mkdir(exist_ok=True)
    title = card([(TITLE_CARD[0], TEXT), (TITLE_CARD[1], NAMED["cyan"]), (TITLE_CARD[2], (139, 148, 158))],
                 [TITLE, SUBTITLE, SUBTITLE])
    results = card([(t, NAMED["green"] if bold else TEXT) for t, bold in RESULTS_CARD],
                   [CARD_BOLD if bold else CARD for _, bold in RESULTS_CARD])
    # Faster video: every screen is shown `speed` times shorter, but the two cards never less
    # than 3 and 5 seconds, so they can still be read.
    timeline = ([(title, max(3.0, 4.0 / speed))] + [(screen(s), sec / speed) for s, sec in shots]
                + [(results, max(5.0, 7.0 / speed))])
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as video:
        for frame, seconds in timeline:
            for _ in range(max(1, round(seconds * FPS))):
                video.append_data(frame)
    total = sum(seconds for _, seconds in timeline)
    print(f"Saved {out} ({total:.0f} seconds, {len(timeline)} screens)")
    return timeline                             # (frame, seconds) pairs, for timing a narration


if __name__ == "__main__":
    main()
