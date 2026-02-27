"""
UI Element Data Collection & Labeling Tool.
Takes screenshots and lets you draw bounding boxes around UI elements,
then saves annotations in YOLO-like format for training.

Usage:
    python ml_model/collect_data.py

Controls:
    - Press 'c' to capture a new screenshot
    - Draw bounding boxes by clicking and dragging
    - Press 1-9 to assign class label to the last drawn box
    - Press 'u' to undo last box
    - Press 's' to save current annotations
    - Press 'n' for next (save + new capture)
    - Press 'q' to quit

Classes:
    1: button        2: search_bar    3: text_field
    4: video_thumb   5: link          6: icon
    7: tab           8: address_bar   9: menu_item
"""

import os
import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Class definitions ───────────────────────────────────────────────

CLASSES = {
    1: "button",
    2: "search_bar",
    3: "text_field",
    4: "video_thumbnail",
    5: "link",
    6: "icon",
    7: "tab",
    8: "address_bar",
    9: "menu_item",
}

CLASS_COLORS = {
    "button":          (0, 255, 0),      # Green
    "search_bar":      (255, 165, 0),    # Orange
    "text_field":      (0, 255, 255),    # Yellow
    "video_thumbnail": (255, 0, 255),    # Magenta
    "link":            (255, 0, 0),      # Blue
    "icon":            (0, 165, 255),    # Orange-red
    "tab":             (128, 0, 128),    # Purple
    "address_bar":     (0, 128, 255),    # Light orange
    "menu_item":       (128, 128, 0),    # Teal
}


# ── Data directories ───────────────────────────────────────────────

DATA_DIR = Path("ml_model/data")
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Screenshot capture ─────────────────────────────────────────────

def capture_screenshot() -> np.ndarray:
    """Capture the current screen using mss (fast)."""
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        # Convert BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ── Labeling GUI ───────────────────────────────────────────────────

class LabelingTool:
    """Interactive bounding box labeling tool using OpenCV."""

    def __init__(self):
        self.boxes = []          # List of (x1, y1, x2, y2, class_name)
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0
        self.image = None
        self.display_image = None
        self.scale = 1.0         # Scale factor for display
        self.image_count = self._count_existing_images()

    def _count_existing_images(self) -> int:
        """Count existing images to continue numbering."""
        return len(list(IMAGES_DIR.glob("*.png")))

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        # Scale back to original image coordinates
        ox = int(x / self.scale)
        oy = int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = ox
            self.start_y = oy
            self.current_x = ox
            self.current_y = oy

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x = ox
            self.current_y = oy

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1 = min(self.start_x, ox)
            y1 = min(self.start_y, oy)
            x2 = max(self.start_x, ox)
            y2 = max(self.start_y, oy)

            # Minimum box size
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                # Add box with "unlabeled" class — user presses 1-9 to assign
                self.boxes.append((x1, y1, x2, y2, "unlabeled"))
                print(f"  Box drawn: ({x1},{y1}) → ({x2},{y2})  |  Press 1-9 to label it")
            else:
                print("  Box too small, ignored.")

    def _draw_boxes(self, img):
        """Draw all bounding boxes on the image."""
        overlay = img.copy()
        for (x1, y1, x2, y2, cls_name) in self.boxes:
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            sx1, sy1 = int(x1 * self.scale), int(y1 * self.scale)
            sx2, sy2 = int(x2 * self.scale), int(y2 * self.scale)
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, 2)
            label = cls_name.replace("_", " ").title()
            cv2.putText(overlay, label, (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw current box being drawn
        if self.drawing:
            sx1, sy1 = int(self.start_x * self.scale), int(self.start_y * self.scale)
            sx2, sy2 = int(self.current_x * self.scale), int(self.current_y * self.scale)
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)

        return overlay

    def _save_annotation(self, image, filename_stem):
        """Save image and labels in YOLO-like format."""
        # Save image
        img_path = IMAGES_DIR / f"{filename_stem}.png"
        cv2.imwrite(str(img_path), image)

        # Save labels: class_index cx cy w h (normalized)
        h_img, w_img = image.shape[:2]
        label_path = LABELS_DIR / f"{filename_stem}.txt"

        # Also save as JSON for easier loading during training
        json_path = LABELS_DIR / f"{filename_stem}.json"
        json_data = []

        with open(label_path, "w") as f:
            for (x1, y1, x2, y2, cls_name) in self.boxes:
                if cls_name == "unlabeled":
                    continue

                # Find class index
                cls_idx = -1
                for k, v in CLASSES.items():
                    if v == cls_name:
                        cls_idx = k - 1  # 0-indexed
                        break
                if cls_idx < 0:
                    continue

                # Normalized YOLO format
                cx = ((x1 + x2) / 2) / w_img
                cy = ((y1 + y2) / 2) / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                json_data.append({
                    "class": cls_name,
                    "class_idx": cls_idx,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_normalized": [cx, cy, bw, bh],
                })

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        labeled = sum(1 for b in self.boxes if b[4] != "unlabeled")
        print(f"  💾 Saved: {img_path.name} with {labeled} labeled boxes")

    def run(self):
        """Main labeling loop."""
        print("\n" + "=" * 60)
        print("  UI Element Data Collection & Labeling Tool")
        print("=" * 60)
        print(f"  Existing images: {self.image_count}")
        print(f"  Save directory:  {DATA_DIR.absolute()}")
        print()
        print("  Controls:")
        print("    c = Capture screenshot")
        print("    Draw boxes with mouse click+drag")
        print("    1-9 = Assign class to last box")
        print("    u = Undo last box")
        print("    s = Save current annotations")
        print("    n = Save + capture next")
        print("    q = Quit")
        print()
        print("  Classes:")
        for k, v in CLASSES.items():
            print(f"    {k}: {v}")
        print("=" * 60)

        cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Labeling Tool", self._mouse_callback)

        # Capture first screenshot
        print("\n  Press 'c' to capture first screenshot...")

        # Show blank placeholder
        placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Press 'c' to capture screenshot",
                    (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Labeling Tool", placeholder)

        while True:
            # Display
            if self.image is not None:
                display = cv2.resize(self.image, None,
                                     fx=self.scale, fy=self.scale)
                display = self._draw_boxes(display)

                # Status bar
                h, w = display.shape[:2]
                bar = np.zeros((40, w, 3), dtype=np.uint8)
                n_boxes = len(self.boxes)
                n_labeled = sum(1 for b in self.boxes if b[4] != "unlabeled")
                status = f"Boxes: {n_boxes} | Labeled: {n_labeled} | Image #{self.image_count}"
                cv2.putText(bar, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                display = np.vstack([display, bar])

                cv2.imshow("Labeling Tool", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n  Quitting...")
                break

            elif key == ord('c'):
                # Capture new screenshot
                print("  📸 Capturing screenshot...")
                self.image = capture_screenshot()
                self.boxes = []
                h, w = self.image.shape[:2]
                # Scale to fit screen (max 1280 wide)
                self.scale = min(1.0, 1280 / w)
                print(f"  Screenshot: {w}x{h} (display scale: {self.scale:.2f})")

            elif key == ord('s'):
                # Save current
                if self.image is not None:
                    stem = f"ui_{self.image_count:04d}"
                    self._save_annotation(self.image, stem)

            elif key == ord('n'):
                # Save + capture next
                if self.image is not None:
                    stem = f"ui_{self.image_count:04d}"
                    self._save_annotation(self.image, stem)
                    self.image_count += 1

                print("  📸 Capturing next screenshot...")
                self.image = capture_screenshot()
                self.boxes = []
                h, w = self.image.shape[:2]
                self.scale = min(1.0, 1280 / w)

            elif key == ord('u'):
                # Undo last box
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"  ↩ Undid box: {removed[4]}")
                else:
                    print("  No boxes to undo.")

            elif ord('1') <= key <= ord('9'):
                # Assign class to last unlabeled box
                class_num = key - ord('0')
                if class_num in CLASSES:
                    # Find last unlabeled box, or just relabel the last box
                    for i in range(len(self.boxes) - 1, -1, -1):
                        if self.boxes[i][4] == "unlabeled":
                            x1, y1, x2, y2, _ = self.boxes[i]
                            self.boxes[i] = (x1, y1, x2, y2, CLASSES[class_num])
                            print(f"  ✅ Labeled as: {CLASSES[class_num]}")
                            break
                    else:
                        # No unlabeled box, relabel the last one
                        if self.boxes:
                            x1, y1, x2, y2, _ = self.boxes[-1]
                            self.boxes[-1] = (x1, y1, x2, y2, CLASSES[class_num])
                            print(f"  ✅ Relabeled last box as: {CLASSES[class_num]}")

        cv2.destroyAllWindows()
        print(f"\n  Done! Total images in dataset: {self.image_count}")
        print(f"  Data saved to: {DATA_DIR.absolute()}\n")


# ── Auto-augmentation ──────────────────────────────────────────────

def augment_dataset():
    """Apply basic augmentations to boost dataset size (offline alternative to Roboflow)."""
    print("\n  Augmenting dataset...")
    aug_dir = DATA_DIR / "augmented_images"
    aug_label_dir = DATA_DIR / "augmented_labels"
    aug_dir.mkdir(exist_ok=True)
    aug_label_dir.mkdir(exist_ok=True)

    count = 0
    for img_path in IMAGES_DIR.glob("*.png"):
        img = cv2.imread(str(img_path))
        stem = img_path.stem
        label_path = LABELS_DIR / f"{stem}.json"

        if not label_path.exists():
            continue

        with open(label_path) as f:
            labels = json.load(f)

        h, w = img.shape[:2]

        # Augmentation 1: Brightness increase
        bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        aug_stem = f"{stem}_bright"
        cv2.imwrite(str(aug_dir / f"{aug_stem}.png"), bright)
        with open(aug_label_dir / f"{aug_stem}.json", "w") as f:
            json.dump(labels, f, indent=2)
        count += 1

        # Augmentation 2: Brightness decrease
        dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
        aug_stem = f"{stem}_dark"
        cv2.imwrite(str(aug_dir / f"{aug_stem}.png"), dark)
        with open(aug_label_dir / f"{aug_stem}.json", "w") as f:
            json.dump(labels, f, indent=2)
        count += 1

        # Augmentation 3: Horizontal flip (adjust bbox x-coordinates)
        flipped = cv2.flip(img, 1)
        flipped_labels = []
        for lbl in labels:
            x1, y1, x2, y2 = lbl["bbox"]
            new_x1 = w - x2
            new_x2 = w - x1
            flipped_labels.append({
                **lbl,
                "bbox": [new_x1, y1, new_x2, y2],
            })
        aug_stem = f"{stem}_flip"
        cv2.imwrite(str(aug_dir / f"{aug_stem}.png"), flipped)
        with open(aug_label_dir / f"{aug_stem}.json", "w") as f:
            json.dump(flipped_labels, f, indent=2)
        count += 1

        # Augmentation 4: Slight Gaussian blur
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        aug_stem = f"{stem}_blur"
        cv2.imwrite(str(aug_dir / f"{aug_stem}.png"), blurred)
        with open(aug_label_dir / f"{aug_stem}.json", "w") as f:
            json.dump(labels, f, indent=2)
        count += 1

    print(f"  Created {count} augmented images")
    print(f"  Saved to: {aug_dir.absolute()}")


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UI Element Data Collection Tool")
    parser.add_argument("--augment", action="store_true", help="Run augmentation on existing data")
    args = parser.parse_args()

    if args.augment:
        augment_dataset()
    else:
        tool = LabelingTool()
        tool.run()
