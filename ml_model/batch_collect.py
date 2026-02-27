"""
Batch Screenshot Collector — automatically navigates to various webpages
and captures screenshots for labeling.

This creates a diverse dataset of UI screenshots for training the RF model.

Usage:
    python ml_model/batch_collect.py              # Capture all pages
    python ml_model/batch_collect.py --start 4    # Start numbering from ui_0004
    python ml_model/batch_collect.py --delay 3    # 3 second delay per page

After running this, label the screenshots with:
    python ml_model/collect_data.py
"""

import os
import sys
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

# Contributor prefix — set CONTRIBUTOR in .env (e.g. CONTRIBUTOR=alice).
# This prevents filename collisions when multiple people collect data.
CONTRIBUTOR = os.getenv("CONTRIBUTOR", "shared").strip().lower().replace(" ", "_")

DATA_DIR = Path("ml_model/data")
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def capture_screenshot() -> np.ndarray:
    """Capture current screen."""
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def count_existing(prefix: str) -> int:
    """Count existing images for this contributor only."""
    return len(list(IMAGES_DIR.glob(f"{prefix}_ui_*.png")))


# Pages to visit for diverse UI screenshots
PAGES = [
    # (url_or_action, description, wait_seconds)
    ("https://www.youtube.com", "YouTube home", 4),
    ("https://www.youtube.com/results?search_query=python+tutorial", "YouTube search results", 4),
    ("https://www.google.com", "Google home", 3),
    ("https://www.google.com/search?q=machine+learning", "Google search results", 3),
    ("https://www.github.com", "GitHub home", 4),
    ("https://www.wikipedia.org", "Wikipedia home", 3),
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "Wikipedia article", 3),
    ("https://www.reddit.com", "Reddit home", 4),
    ("chrome://newtab", "Chrome new tab", 2),
    ("https://www.bing.com", "Bing home", 3),
    ("https://www.amazon.com", "Amazon home", 4),
    ("https://www.twitter.com", "Twitter/X home", 4),
    ("https://www.stackoverflow.com", "StackOverflow home", 4),
    ("https://www.netflix.com", "Netflix home", 4),
    ("https://mail.google.com", "Gmail", 4),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Screenshot Collector")
    parser.add_argument("--start", type=int, default=None,
                        help="Start image numbering from this index")
    parser.add_argument("--delay", type=float, default=None,
                        help="Override wait time per page (seconds)")
    parser.add_argument("--pages", type=int, default=len(PAGES),
                        help=f"Number of pages to capture (max {len(PAGES)})")
    args = parser.parse_args()

    prefix = CONTRIBUTOR
    idx = args.start if args.start is not None else count_existing(prefix) + 1
    n_pages = min(args.pages, len(PAGES))

    print("\n" + "=" * 60)
    print("  Batch Screenshot Collector")
    print("=" * 60)
    print(f"  Contributor: {prefix}")
    print(f"  Will capture {n_pages} pages")
    print(f"  Starting at: {prefix}_ui_{idx:04d}")
    print(f"  Save to: {IMAGES_DIR.absolute()}")
    print()
    print("  ⚠️  Make sure Chrome is open and visible!")
    print("  Starting in 3 seconds...")
    time.sleep(3)

    import pyautogui
    captured = 0

    for i, (url, desc, wait) in enumerate(PAGES[:n_pages]):
        if args.delay is not None:
            wait = args.delay

        print(f"\n  [{i+1}/{n_pages}] {desc}")
        print(f"    Navigating to: {url}")

        # Focus Chrome address bar and navigate
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.write(url, interval=0.01)
        pyautogui.press("enter")
        time.sleep(wait)

        # Capture screenshot
        frame = capture_screenshot()
        filename = f"{prefix}_ui_{idx:04d}.png"
        filepath = IMAGES_DIR / filename
        cv2.imwrite(str(filepath), frame)
        print(f"    📸 Saved: {filename} ({frame.shape[1]}x{frame.shape[0]})")

        idx += 1
        captured += 1

    print(f"\n  ✅ Captured {captured} screenshots!")
    print(f"  Images saved to: {IMAGES_DIR.absolute()}")
    print()
    print("  Next step: Label the screenshots with:")
    print("    python ml_model/collect_data.py")
    print()
    print("  Then retrain:")
    print("    python ml_model/train_rf.py --augment")
    print()


if __name__ == "__main__":
    main()
