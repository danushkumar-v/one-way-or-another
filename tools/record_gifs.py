"""Record the README GIFs and stills from the deck (needs: pip install playwright pillow; playwright install chromium).

    python tools/record_gifs.py              # all clips -> assets/
    python tools/record_gifs.py --shots      # stills of every scene only -> assets/shots/ (quick visual check)
"""
import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "index.html"
OUT = ROOT / "assets"
W, H = 1280, 720

# name: (scene, start delay ms, duration ms, fps, gif width)
CLIPS = {
    "city_lights_up": ("lights", 1800, 13500, 10, 880),
    "sign_to_direction": ("hero", 4800, 9500, 10, 880),
    "hidden_areas": ("split", 500, 9000, 8, 760),
    "fuse_signs": ("fuse", 3200, 9000, 9, 760),
    "blocked_turn": ("restrict", 3300, 5000, 9, 760),
    "finale": ("finale", 200, 7500, 10, 880),
}


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"], **kw)
        except Exception:
            continue
    raise SystemExit("no Chromium found: run `playwright install chromium`")


def open_deck(b):
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
    pg.goto(DECK.as_uri())
    pg.wait_for_function("window.deck && window.deck.order", timeout=60000)
    pg.wait_for_timeout(2500)
    pg.add_style_tag(content="#nav,#fps{display:none!important}")
    return pg, errs


def to_gif(frames, path, width, fps):
    imgs = []
    for png in frames:
        im = Image.open(io.BytesIO(png)).convert("RGB")
        im = im.resize((width, round(width * H / W)), Image.LANCZOS)
        imgs.append(im)
    pal = imgs[len(imgs) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0, optimize=True, disposal=1)
    return path.stat().st_size / 1e6


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = launch(p)
        pg, errs = open_deck(b)
        order = pg.evaluate("deck.order")
        if "--shots" in sys.argv:
            (OUT / "shots").mkdir(exist_ok=True)
            for i, name in enumerate(order):
                pg.evaluate(f"deck.go({i})")
                pg.wait_for_timeout({"hero": 15000, "lights": 20000, "fuse": 11500, "restrict": 5200, "finale": 6500}.get(name, 3500))
                pg.screenshot(path=str(OUT / "shots" / f"{i:02d}_{name}.png"))
                print(name, "fps", round(pg.evaluate("deck.fps")))
            print("errors:", errs)
            return
        for name, (scene, delay, dur, fps, width) in CLIPS.items():
            if scene not in order:
                continue
            pg.evaluate("deck.go(0)")
            pg.wait_for_timeout(600)
            pg.evaluate(f"deck.go({order.index(scene)})")
            pg.wait_for_timeout(delay)
            frames, t0, step = [], time.perf_counter(), 1 / fps
            while time.perf_counter() - t0 < dur / 1000:
                t = time.perf_counter()
                frames.append(pg.screenshot(type="png"))
                time.sleep(max(0, step - (time.perf_counter() - t)))
            mb = to_gif(frames, OUT / f"{name}.gif", width, fps)
            print(f"{name}.gif  {len(frames)} frames  {mb:.1f} MB")
        for scene in ("title", "models", "proof", "targets"):
            pg.evaluate(f"deck.go({order.index(scene)})")
            pg.wait_for_timeout(3800)
            pg.screenshot(path=str(OUT / f"slide_{scene}.png"))
        print("errors:", errs)
        b.close()


if __name__ == "__main__":
    main()
