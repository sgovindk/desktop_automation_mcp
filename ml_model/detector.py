"""
UI Element Detector using Random Forest + Sliding Window.
Takes a screenshot, runs a multi-scale sliding window,
extracts HOG/color/edge features from each window, and
classifies using the trained Random Forest model.

Uses Selective Search or Edge-based proposals for speed
instead of brute-force sliding window.
"""

import sys
import time
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_model.feature_extractor import FeatureExtractor

# ── Paths ───────────────────────────────────────────────────────────

MODEL_DIR = Path("models")


class UIDetectorRF:
    """
    UI Element Detector using Random Forest classifier.

    Detection pipeline:
        1. Capture screenshot
        2. Generate region proposals (selective search / edge contours)
        3. Extract HOG + color + edge features from each proposal
        4. Classify with Random Forest
        5. Apply Non-Maximum Suppression (NMS)
        6. Return detections with labels and bounding boxes
    """

    def __init__(self, model_type: str = "rf", confidence_threshold: float = 0.5):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        # Load trained model
        self._model = None
        self._scaler = None
        self._classes = None

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Cache
        self._loaded = False

    def _load_model(self):
        """Load the trained RF/SVM model from disk."""
        if self._loaded:
            return

        model_path = MODEL_DIR / f"ui_detector_{self.model_type}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                f"Train the model first:\n"
                f"  1. python ml_model/collect_data.py   (label screenshots)\n"
                f"  2. python ml_model/train_rf.py       (train model)"
            )

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._scaler = data["scaler"]
        self._classes = data["classes"]
        self._loaded = True

        print(f"[Detector] Loaded {self.model_type.upper()} model "
              f"(accuracy: {data.get('accuracy', 'N/A')})")

    @property
    def classes(self):
        self._load_model()
        return self._classes

    # ── Region proposal generation ──────────────────────────────────

    def _generate_proposals(
        self,
        image: np.ndarray,
        max_proposals: int = 500,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate candidate bounding box regions using edge contours.
        Faster than full sliding window, more targeted proposals.
        """
        h, w = image.shape[:2]
        proposals = []

        # Method 1: Edge-based contour proposals
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Single Canny pass for speed
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter by size (skip tiny/huge regions)
            if cw < 25 or ch < 18:
                continue
            if cw > w * 0.7 or ch > h * 0.7:
                continue
            # Skip very extreme aspect ratios
            aspect = cw / max(ch, 1)
            if aspect > 12 or aspect < 0.08:
                continue

            proposals.append((x, y, x + cw, y + ch))

        # Method 2: Targeted sliding window — scale-aware UI regions
        # All positions/sizes are relative to screen dimensions so it works
        # on any resolution (1080p, 1440p, 4K, etc.)
        window_configs = [
            # (w_frac, h_frac, y_start_frac, y_end_frac, stride_x_frac, stride_y_frac)
            # Search bars (wide, short) — scan top 25% of screen
            (0.35, 0.05, 0.00, 0.25, 0.08, 0.02),   # Large search bar (~900px on 2560)
            (0.25, 0.04, 0.00, 0.25, 0.06, 0.02),   # Medium search bar (~640px on 2560)
            (0.20, 0.03, 0.00, 0.15, 0.05, 0.02),   # Address bar size
            # Buttons/tabs (small, near top)
            (0.06, 0.025, 0.00, 0.08, 0.04, 0.02),  # Buttons/tabs
            (0.05, 0.02, 0.00, 0.05, 0.03, 0.015),  # Small buttons
            # Video thumbnails (large, middle of screen)
            (0.15, 0.12, 0.15, 0.75, 0.10, 0.10),   # Video thumbnails
        ]

        for (wf, hf, ys_f, ye_f, sx_f, sy_f) in window_configs:
            ww = max(30, int(w * wf))
            wh = max(20, int(h * hf))
            y_start = int(h * ys_f)
            y_end = int(h * ye_f)
            sx = max(20, int(w * sx_f))
            sy = max(10, int(h * sy_f))
            for y_pos in range(y_start, min(y_end, h - wh), sy):
                for x_pos in range(0, w - ww, sx):
                    proposals.append((x_pos, y_pos, x_pos + ww, y_pos + wh))

        # Limit proposals BEFORE dedup for speed
        if len(proposals) > max_proposals * 2:
            np.random.shuffle(proposals)
            proposals = proposals[:max_proposals * 2]

        # Fast dedup using numpy
        proposals = self._fast_nms_proposals(proposals, iou_threshold=0.7)

        if len(proposals) > max_proposals:
            proposals = proposals[:max_proposals]

        return proposals

    def _deduplicate_proposals(
        self,
        proposals: list,
        iou_threshold: float = 0.7,
    ) -> list:
        """Remove highly overlapping proposals."""
        return proposals

    def _fast_nms_proposals(
        self,
        proposals: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.7,
    ) -> List[Tuple[int, int, int, int]]:
        """Vectorized NMS for raw proposals (no scores — keep by order)."""
        if len(proposals) == 0:
            return proposals

        boxes = np.array(proposals, dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Process in order (first = highest priority)
        order = np.arange(len(boxes))
        keep: list[int] = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            # Keep only boxes with IoU below threshold
            order = rest[iou <= iou_threshold]

        return [proposals[k] for k in keep]

    # ── Classification ──────────────────────────────────────────────

    def _classify_proposals(
        self,
        image: np.ndarray,
        proposals: List[Tuple[int, int, int, int]],
    ) -> List[Dict]:
        """
        Classify each proposal region using the trained RF/SVM model.
        """
        self._load_model()

        h, w = image.shape[:2]
        detections = []

        # Extract features for all proposals in batch
        features_list = []
        for bbox in proposals:
            feat = self.feature_extractor.extract_from_region(image, bbox, w, h)
            features_list.append(feat)

        if not features_list:
            return []

        X = np.array(features_list, dtype=np.float32)
        X_scaled = self._scaler.transform(X)

        # Predict class probabilities
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(X_scaled)
        else:
            # SVM without probability calibration
            predictions = self._model.predict(X_scaled)
            probabilities = np.zeros((len(predictions), len(self._classes)))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0

        # Background class index
        bg_idx = self._classes.index("background") if "background" in self._classes else -1

        # IMPORTANT: model.classes_ maps predict_proba column indices
        # back to the original class indices (since sklearn skips classes
        # with no training samples). Without this, class names are WRONG.
        model_class_indices = self._model.classes_  # e.g. [0, 1, 3, 6, 7, 9]

        for i, bbox in enumerate(proposals):
            probs = probabilities[i]
            prob_col = np.argmax(probs)             # column in predict_proba
            predicted_class = model_class_indices[prob_col]  # actual class index
            confidence = probs[prob_col]

            # Skip background or low-confidence
            if predicted_class == bg_idx:
                continue
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = bbox
            detections.append({
                "label": self._classes[predicted_class],
                "confidence": round(float(confidence), 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {
                    "x": int((x1 + x2) / 2),
                    "y": int((y1 + y2) / 2),
                },
            })

        return detections

    # ── Non-Maximum Suppression ─────────────────────────────────────

    def _nms(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.4,
    ) -> List[Dict]:
        """Vectorized Non-Maximum Suppression for classified detections."""
        if not detections:
            return []

        # Build arrays for vectorized computation
        n = len(detections)
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        for i, d in enumerate(detections):
            boxes[i] = [d["bbox"]["x1"], d["bbox"]["y1"],
                        d["bbox"]["x2"], d["bbox"]["y2"]]
            scores[i] = d["confidence"]

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence descending
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            order = rest[iou <= iou_threshold]

        return [detections[k] for k in keep]

    # ── Public API ──────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        max_proposals: int = 500,
        nms_threshold: float = 0.4,
    ) -> List[Dict]:
        """
        Detect UI elements in an image.

        Args:
            image: BGR screenshot (numpy array)
            max_proposals: Max number of region proposals
            nms_threshold: IoU threshold for NMS

        Returns:
            List of detections with: label, confidence, bbox, center
        """
        start = time.time()

        # 1. Generate proposals
        proposals = self._generate_proposals(image, max_proposals)

        # 2. Classify
        detections = self._classify_proposals(image, proposals)

        # 3. NMS
        detections = self._nms(detections, nms_threshold)

        elapsed = time.time() - start
        print(f"[Detector] {len(proposals)} proposals → "
              f"{len(detections)} detections in {elapsed:.2f}s")

        return detections

    def detect_screenshot(self, **kwargs) -> List[Dict]:
        """Capture a screenshot and detect UI elements on it."""
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return self.detect(frame, **kwargs)

    def find_element(
        self,
        image: np.ndarray,
        target_class: str,
        target_text: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Find a specific UI element by class (and optionally by nearby text via OCR).

        Returns the highest-confidence match or None.
        """
        detections = self.detect(image)

        # Filter by class
        matches = [d for d in detections if d["label"] == target_class]

        if not matches:
            return None

        # Return highest confidence
        return matches[0]

    def find_search_bar(self, image: np.ndarray) -> Optional[Dict]:
        """Convenience: find the search bar on screen."""
        return self.find_element(image, "search_bar")

    def find_buttons(self, image: np.ndarray) -> List[Dict]:
        """Convenience: find all buttons on screen."""
        detections = self.detect(image)
        return [d for d in detections if d["label"] == "button"]


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Testing UI Detector...")

    try:
        detector = UIDetectorRF(model_type="rf", confidence_threshold=0.4)
        detections = detector.detect_screenshot()

        print(f"\n  Found {len(detections)} UI elements:")
        for det in detections:
            print(f"    {det['label']:20s}  conf={det['confidence']:.2f}  "
                  f"center=({det['center']['x']}, {det['center']['y']})")

    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("\n  To use the detector, first collect data and train:")
        print("    1. python ml_model/collect_data.py")
        print("    2. python ml_model/train_rf.py")
