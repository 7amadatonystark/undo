"""
make_code_video.py — a video tour of the code: the parts of UNDO and the path one action takes.

It draws a diagram of the parts, then shows one slide per part with REAL code, read from the
files when the video is built (so the video always matches the code), coloured like an editor,
with a one-line caption. No API calls.

    python make_code_video.py              # media/undo_code_tour.mp4 (about 2 minutes)
    python make_code_video.py --speed 2    # media/undo_code_tour_x2.mp4: twice as fast
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String

from make_video import (BACKGROUND, CARD, CARD_BOLD, FONTS, FPS, HEIGHT, MONO, MONO_BOLD,
                        NAMED, RESULTS_CARD, SUBTITLE, TEXT, TITLE, WIDTH, card)

ROOT = Path(__file__).parent
OUT = ROOT / "media" / "undo_code_tour.mp4"
GREY = (139, 148, 158)
PANEL = (22, 27, 34)
BORDER = (48, 54, 61)
HEADING = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 46)
CAPTION = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 34)
SMALL = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 26)
BOX = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 32)
CODE_LINE = 34
MAX_CODE_LINES = 22

# Editor colours for each kind of Python token (the first match wins).
TOKEN_COLORS = [(Comment, GREY), (String, (165, 214, 255)), (Number, (121, 192, 255)),
                (Keyword, (255, 123, 114)), (Name.Function, (210, 168, 255)),
                (Name.Class, (210, 168, 255)), (Name.Builtin, (255, 166, 87)),
                (Name.Decorator, (255, 166, 87)), (Operator, (255, 123, 114))]

# One slide per part, in the order an action meets them:
# (part, file, [(first line, last line), ...], caption). A first/last line is found by the
# text it contains; several pieces are joined with "...".
SLIDES = [
    ("The AI agent", "agent.py",
     [("def agent_loop(", "steps = []"), ("while True:", "return steps, (reply.content"),
      ("for i, call in enumerate", "for i, call in enumerate"),
      ("result = call_tool(call, sandbox, runtime)", "result = call_tool(call, sandbox, runtime)")],
     "The model picks a tool. Every tool call goes to UNDO, never straight to the world."),
    ("Every action knows its undo", "undo/actions.py",
     [("class DeleteRows(Action):", "kind = REVERSIBLE"), ("def prepare(self):", "con.close()"),
      ("def do(self):", 'con.execute(f"DELETE'), ("def undo(self):", "self.saved_rows)")],
     "prepare() saves the rows, do() deletes them, undo() puts them back."),
    ("1. Classify this call", "undo/classifier.py",
     [("def classify(action, sandbox):", "return kind, reasons")],
     "Each call gets a kind. A rule can make it stricter, like an email that carries a password."),
    ("2. Gate, 3. stage, write-ahead journal", "undo/runtime.py",
     [("# 2. gate.", "t0 = time.perf_counter()"), ("# 3. stage:", "action.prepare()"),
      ("# Write-ahead:", 'self.journal({"event": "do_begin"'),
      ("action.do()             # the real effect", "action.do()             # the real effect")],
     "Irreversible? A person decides first. Then save the undo, write it down, and only then do it."),
    ("What changed, in plain words", "undo/changes.py",
     [("def summarize(before, after):", 'return "\\n".join')],
     "The verifier never sees the answer key: only the task and this list of changes."),
    ("The verifier", "task_verifiers.py",
     [("def combine(rule_result, llm_result):", "return rules_ok and llm_ok"),
      ("def combined(task, changes", "return combine(rule_verifier")],
     "Simple rules and an AI judge. If either says NOT OK, the work is rolled back."),
    ("Roll back, newest first", "undo/runtime.py",
     [("def rollback(self):", "self.log(f\"       undid:")],
     "Undo in reverse order. One failed undo never stops the others."),
    ("Crash recovery", "undo/runtime.py",
     [("def recover(sandbox, verbose=False):", "return 0, []")],
     "If the program dies mid-task, the journal tells recover() what to undo on restart."),
    ("A benchmark task from a real incident", "scenarios.py",
     [('trap("replit_code_freeze"', "or not all(file_unchanged")],
     "The answer key (damaged) is used only for SCORING. UNDO's decisions never read it."),
    ("Statistics", "analyze.py",
     [("def wilson(k, n, z=1.96):", "return (max(0.0, centre - half)")],
     "Every rate gets a 95% interval, and Fisher's exact test checks it is not luck."),
]


def excerpt(file, pieces):
    """The lines of `file` from each (first, last) marker pair, joined with "..." lines. Each
    piece is searched for after the previous one, so a common line like "def do(self):" is
    found in the right class."""
    lines = (ROOT / file).read_text(encoding="utf-8").splitlines()
    out, end = [], -1
    for first, last in pieces:
        start = next(i for i in range(end + 1, len(lines)) if first in lines[i])
        end = next(i for i in range(start, len(lines)) if last in lines[i])
        if out:
            out.append("...")
        out += lines[start:end + 1]
    indent = min(len(l) - len(l.lstrip()) for l in out if l.strip() and l != "...")
    out = [l[indent:] if l != "..." else l for l in out]
    if len(out) > MAX_CODE_LINES:
        raise ValueError(f"{file}: {len(out)} lines is too many for one slide")
    return out


def color_of(token):
    for kind, color in TOKEN_COLORS:
        if token in kind:
            return color
    return TEXT


def code_slide(number, part, file, code, caption):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((120, 50), f"{number} / {len(SLIDES)}", font=SMALL, fill=GREY)
    draw.text((120, 80), part, font=HEADING, fill=TEXT)
    draw.text((WIDTH - 120 - MONO.getlength(file), 95), file, font=MONO, fill=NAMED["cyan"])
    top = 170
    draw.rounded_rectangle((100, top, WIDTH - 100, top + 40 + MAX_CODE_LINES * CODE_LINE),
                           radius=14, fill=PANEL, outline=BORDER, width=2)
    char = MONO.getlength("M")
    y = top + 20 + (MAX_CODE_LINES - len(code)) * CODE_LINE // 2       # centred in the panel
    for line in code:
        x = 140
        if line == "...":
            draw.text((x, y), "...", font=MONO, fill=GREY)
        else:
            for token, text in PythonLexer().get_tokens(line):
                text = text.rstrip("\n")
                draw.text((x, y), text, font=MONO, fill=color_of(token))
                x += len(text) * char
        y += CODE_LINE
    width = draw.textlength(caption, font=CAPTION)
    draw.text(((WIDTH - width) / 2, HEIGHT - 105), caption, font=CAPTION, fill=NAMED["yellow"])
    return np.asarray(image)


def box(draw, xy, title, file, color, fill=PANEL):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=color, width=4)
    x0, y0, x1, _ = xy
    w = draw.textlength(title, font=BOX)
    draw.text(((x0 + x1 - w) / 2, y0 + 22), title, font=BOX, fill=TEXT)
    w = MONO_BOLD.getlength(file)
    draw.text(((x0 + x1 - w) / 2, y0 + 70), file, font=MONO, fill=color)


def arrow(draw, start, end, label, color=GREY, label_at=None):
    draw.line((start, end), fill=color, width=5)
    (x0, y0), (x1, y1) = start, end
    length = max(1, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    draw.polygon([end, (x1 - 24 * ux + 12 * uy, y1 - 24 * uy - 12 * ux),
                  (x1 - 24 * ux - 12 * uy, y1 - 24 * uy + 12 * ux)], fill=color)
    lx, ly = label_at or ((x0 + x1) / 2, (y0 + y1) / 2 - 42)
    w = draw.textlength(label, font=SMALL)
    draw.text((lx - w / 2, ly), label, font=SMALL, fill=color)


def diagram():
    """The parts of UNDO and the path of one action (numbered like the code slides)."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = "The parts of UNDO, and the path of one action"
    draw.text(((WIDTH - draw.textlength(title, font=HEADING)) / 2, 50), title, font=HEADING, fill=TEXT)
    box(draw, (80, 300, 470, 460), "AI agent", "agent.py", NAMED["cyan"])
    box(draw, (1450, 300, 1840, 460), "Fake world", "undo/sandbox.py", NAMED["green"])
    # the runtime, with its steps inside
    draw.rounded_rectangle((640, 200, 1280, 560), radius=18, fill=PANEL, outline=NAMED["yellow"], width=4)
    draw.text((960 - draw.textlength("UNDO runtime", font=BOX) / 2, 222), "UNDO runtime", font=BOX, fill=TEXT)
    draw.text((960 - MONO.getlength("undo/runtime.py") / 2, 268), "undo/runtime.py", font=MONO, fill=NAMED["yellow"])
    steps = [("1  classify the call", "classifier.py"), ("2  gate: ask a person?", ""),
             ("3  stage the undo + journal", "actions.py"), ("4  hold emails and payments", "")]
    for i, (text, file) in enumerate(steps):
        y = 325 + i * 55
        draw.text((690, y), text, font=SMALL, fill=TEXT)
        if file:
            draw.text((1230 - MONO.getlength(file), y + 2), file, font=MONO, fill=GREY)
    box(draw, (640, 700, 1280, 860), "Verifier", "task_verifiers.py", NAMED["red"])
    arrow(draw, (470, 380), (630, 380), "tool call")
    arrow(draw, (1290, 380), (1440, 380), "action")
    arrow(draw, (1645, 470), (1290, 760), "what changed", label_at=(1560, 640))
    arrow(draw, (850, 690), (850, 570), "commit or", label_at=(745, 595))
    draw.text((745 - draw.textlength("roll back", font=SMALL) / 2, 625), "roll back", font=SMALL, fill=GREY)
    note = ("The benchmark (scenarios.py → run_final.py → analyze.py) runs real AI agents "
            "with and without UNDO and scores the damage.")
    draw.text(((WIDTH - draw.textlength(note, font=SMALL)) / 2, 950), note, font=SMALL, fill=GREY)
    return np.asarray(image)


def tests_card():
    count = (ROOT / "test_offline.py").read_text(encoding="utf-8").count("    def test_")
    return card([(f"{count} offline tests", TEXT),
                 ("check every safety rule with a fake AI model,", TEXT),
                 ("so they run in seconds, with no API calls.", TEXT),
                 ("", TEXT),
                 ("python test_offline.py", NAMED["cyan"])],
                [CARD_BOLD, CARD, CARD, CARD, CARD])


def main():
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    out = OUT if speed == 1 else OUT.with_name(f"undo_code_tour_x{speed:g}.mp4")
    title = card([("UNDO", TEXT), ("Inside the code", NAMED["cyan"]),
                  ("The parts, and the path one action takes", GREY)], [TITLE, SUBTITLE, SUBTITLE])
    results = card([(t, NAMED["green"] if bold else TEXT) for t, bold in RESULTS_CARD],
                   [CARD_BOLD if bold else CARD for _, bold in RESULTS_CARD])
    end = card([("Try it yourself", TEXT), ("streamlit run app.py", NAMED["cyan"])], [TITLE, SUBTITLE])
    timeline = [(title, 4.0), (diagram(), 12.0)]
    for n, (part, file, pieces, caption) in enumerate(SLIDES, 1):
        timeline.append((code_slide(n, part, file, excerpt(file, pieces), caption), 10.0))
    timeline += [(tests_card(), 5.0), (results, 7.0), (end, 4.0)]
    # Faster video: every screen is shown `speed` times shorter, but never under 3 seconds.
    timeline = [(frame, max(3.0, seconds / speed)) for frame, seconds in timeline]
    out.parent.mkdir(exist_ok=True)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as video:
        for frame, seconds in timeline:
            for _ in range(round(seconds * FPS)):
                video.append_data(frame)
    total = sum(seconds for _, seconds in timeline)
    print(f"Saved {out} ({total:.0f} seconds, {len(timeline)} screens)")
    return timeline


if __name__ == "__main__":
    main()
