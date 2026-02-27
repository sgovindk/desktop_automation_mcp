"""
Feature Extractor for UI Element Detection.
Extracts HOG (Histogram of Oriented Gradients), color histograms,
edge density, and spatial features from image regions.
These features feed into the Random Forest classifier.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


# ── Constants ───────────────────────────────────────────────────────

# Standard patch size for feature extraction (all regions resized to this)
PATCH_SIZE = (64, 64)

# HOG parameters
HOG_WIN_SIZE = PATCH_SIZE
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector from an image region.

    Feature composition (~500 features total):
        1. HOG features (orientation/shape)        → 324 dims
        2. Color histogram (appearance)             → 96 dims  (32 bins × 3 channels)
        3. Edge features (structure)                → 64 dims
        4. Spatial features (position, aspect ratio)→ 6 dims
        5. Texture (LBP-like statistics)            → 10 dims
    Total ≈ 500 dims
    """

    def __init__(self):
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(
            HOG_WIN_SIZE,
            HOG_BLOCK_SIZE,
            HOG_BLOCK_STRIDE,
            HOG_CELL_SIZE,
            HOG_NBINS,
        )
        self._feature_dim = None

    @property
    def feature_dim(self) -> int:
        """Return total feature dimensionality (computed on first call)."""
        if self._feature_dim is None:
            dummy = np.zeros((*PATCH_SIZE, 3), dtype=np.uint8)
            features = self.extract_from_patch(dummy)
            self._feature_dim = len(features)
        return self._feature_dim

    # ── Main extraction method ──────────────────────────────────────

    def extract_from_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> np.ndarray:
        """
        Extract features from a specific region of an image.

        Args:
            image: Full screenshot (BGR, uint8)
            bbox: (x1, y1, x2, y2) bounding box
            screen_w: Screen width for spatial normalization
            screen_h: Screen height for spatial normalization

        Returns:
            1D float32 feature vector
        """
        x1, y1, x2, y2 = bbox
        # Clamp coordinates
        h_img, w_img = image.shape[:2]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))

        # Crop the region
        patch = image[y1:y2, x1:x2]

        # Extract patch-level features
        patch_features = self.extract_from_patch(patch)

        # Extract spatial features (position on screen)
        spatial = self._spatial_features(x1, y1, x2, y2, screen_w, screen_h)

        return np.concatenate([patch_features, spatial]).astype(np.float32)

    def extract_from_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from an already-cropped image patch.

        Args:
            patch: Cropped image region (BGR, uint8)

        Returns:
            1D float32 feature vector (without spatial features)
        """
        # Resize to standard size
        resized = cv2.resize(patch, PATCH_SIZE)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1) HOG features
        hog_features = self._hog_features(gray)

        # 2) Color histogram
        color_features = self._color_histogram(resized)

        # 3) Edge features
        edge_features = self._edge_features(gray)

        # 4) Texture features
        texture_features = self._texture_features(gray)

        return np.concatenate([
            hog_features,
            color_features,
            edge_features,
            texture_features,
        ]).astype(np.float32)

    # ── Individual feature extractors ───────────────────────────────

    def _hog_features(self, gray: np.ndarray) -> np.ndarray:
        """Histogram of Oriented Gradients — captures shape and edge orientation."""
        hog_vec = self.hog.compute(gray)
        if hog_vec is not None:
            return hog_vec.flatten()
        return np.zeros(324, dtype=np.float32)

    def _color_histogram(self, bgr: np.ndarray, bins: int = 32) -> np.ndarray:
        """Color histogram in HSV space — captures appearance."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        features = []
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        return np.concatenate(features)

    def _edge_features(self, gray: np.ndarray, bins: int = 16) -> np.ndarray:
        """Canny edge density + edge orientation histogram."""
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)

        # Edge density (what fraction of pixels are edges)
        density = np.mean(edges > 0)

        # Sobel gradients for orientation
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Edge orientation histogram
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx) * 180 / np.pi  # -180 to 180
        angle = angle % 360  # 0 to 360

        hist, _ = np.histogram(angle, bins=bins, range=(0, 360), weights=magnitude)
        total = hist.sum()
        if total > 0:
            hist = hist / total

        # Horizontal/vertical edge ratios
        h_edges = np.mean(np.abs(gy) > np.abs(gx))
        v_edges = 1.0 - h_edges

        return np.concatenate([[density, h_edges, v_edges], hist]).astype(np.float32)

    def _texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Simple texture statistics (LBP-like)."""
        # Basic stats
        mean_val = np.mean(gray) / 255.0
        std_val = np.std(gray) / 255.0

        # Laplacian variance (blur detection)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0

        # Local contrast in quadrants
        h, w = gray.shape
        quadrants = [
            gray[:h // 2, :w // 2],     # Top-left
            gray[:h // 2, w // 2:],      # Top-right
            gray[h // 2:, :w // 2],      # Bottom-left
            gray[h // 2:, w // 2:],      # Bottom-right
        ]
        quad_means = [np.mean(q) / 255.0 for q in quadrants]
        quad_stds = [np.std(q) / 255.0 for q in quadrants]

        # Uniformity (how uniform is the region — buttons tend to be uniform)
        uniformity = 1.0 - std_val

        return np.array(
            [mean_val, std_val, lap_var, uniformity] + quad_means + quad_stds,
            dtype=np.float32,
        )[:10]  # cap at 10 features

    def _spatial_features(
        self,
        x1: int, y1: int, x2: int, y2: int,
        screen_w: int, screen_h: int,
    ) -> np.ndarray:
        """Spatial features — where on screen the element is."""
        # Normalized center position
        cx = ((x1 + x2) / 2) / screen_w
        cy = ((y1 + y2) / 2) / screen_h

        # Normalized size
        bw = (x2 - x1) / screen_w
        bh = (y2 - y1) / screen_h

        # Aspect ratio
        aspect = bw / max(bh, 1e-6)

        # Area ratio
        area = bw * bh

        return np.array([cx, cy, bw, bh, aspect, area], dtype=np.float32)

    # ── Batch extraction ────────────────────────────────────────────

    def extract_batch(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Extract features for multiple bounding boxes."""
        h, w = image.shape[:2]
        features = []
        for bbox in bboxes:
            feat = self.extract_from_region(image, bbox, w, h)
            features.append(feat)
        return np.array(features, dtype=np.float32)


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = FeatureExtractor()

    # Test with a dummy image
    dummy = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bbox = (100, 200, 300, 250)

    features = extractor.extract_from_region(dummy, bbox)
    print(f"Feature vector length: {len(features)}")
    print(f"Feature dim property:  {extractor.feature_dim}")
    print(f"Sample features[:10]:  {features[:10]}")
