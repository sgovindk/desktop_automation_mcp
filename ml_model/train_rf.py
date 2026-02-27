"""
Train a Random Forest classifier for UI element detection.
Reads labeled screenshots, extracts HOG/color/edge features,
trains a RandomForest (or SVM), evaluates, and saves the model.

Usage:
    python ml_model/train_rf.py                 # Train Random Forest
    python ml_model/train_rf.py --model svm     # Train SVM instead
    python ml_model/train_rf.py --augment       # Include augmented data
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_model.feature_extractor import FeatureExtractor

# ── Paths ───────────────────────────────────────────────────────────

DATA_DIR = Path("ml_model/data")
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
AUG_IMAGES_DIR = DATA_DIR / "augmented_images"
AUG_LABELS_DIR = DATA_DIR / "augmented_labels"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Classes (must match collect_data.py)
CLASSES = [
    "button", "search_bar", "text_field", "video_thumbnail",
    "link", "icon", "tab", "address_bar", "menu_item",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

# Also add a "background" class for negative samples
CLASSES_WITH_BG = CLASSES + ["background"]
NUM_CLASSES = len(CLASSES_WITH_BG)


# ── Data Loading ────────────────────────────────────────────────────

def load_dataset(include_augmented: bool = False) -> Tuple[List, List]:
    """
    Load images and their JSON annotations.

    Returns:
        samples: List of (image_path, bbox, class_name) tuples
    """
    samples = []

    # Load from main data
    for json_path in LABELS_DIR.glob("*.json"):
        img_path = IMAGES_DIR / f"{json_path.stem}.png"
        if not img_path.exists():
            continue

        with open(json_path) as f:
            annotations = json.load(f)

        for ann in annotations:
            samples.append((str(img_path), ann["bbox"], ann["class"]))

    # Load augmented data
    if include_augmented and AUG_IMAGES_DIR.exists():
        for json_path in AUG_LABELS_DIR.glob("*.json"):
            img_path = AUG_IMAGES_DIR / f"{json_path.stem}.png"
            if not img_path.exists():
                continue

            with open(json_path) as f:
                annotations = json.load(f)

            for ann in annotations:
                samples.append((str(img_path), ann["bbox"], ann["class"]))

    print(f"  Loaded {len(samples)} labeled samples")
    return samples


def generate_negative_samples(
    image: np.ndarray,
    positive_bboxes: List[Tuple[int, int, int, int]],
    num_negatives: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate random negative (background) bounding boxes that don't
    overlap significantly with any positive annotation.
    """
    h, w = image.shape[:2]
    negatives = []
    attempts = 0
    max_attempts = num_negatives * 20

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1

        # Random size
        bw = np.random.randint(30, min(300, w // 2))
        bh = np.random.randint(20, min(200, h // 2))

        # Random position
        x1 = np.random.randint(0, w - bw)
        y1 = np.random.randint(0, h - bh)
        x2 = x1 + bw
        y2 = y1 + bh

        # Check IoU with all positives
        overlaps = False
        for (px1, py1, px2, py2) in positive_bboxes:
            iou = _compute_iou((x1, y1, x2, y2), (px1, py1, px2, py2))
            if iou > 0.2:
                overlaps = True
                break

        if not overlaps:
            negatives.append((x1, y1, x2, y2))

    return negatives


def _compute_iou(box1, box2) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ── Feature extraction pipeline ────────────────────────────────────

def build_feature_matrix(
    samples: list,
    include_negatives: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y from annotated samples.

    Returns:
        X: (n_samples, n_features) float32 array
        y: (n_samples,) int array of class indices
    """
    extractor = FeatureExtractor()

    X_list = []
    y_list = []

    # Group samples by image
    image_groups = {}
    for (img_path, bbox, cls_name) in samples:
        if img_path not in image_groups:
            image_groups[img_path] = []
        image_groups[img_path].append((bbox, cls_name))

    total_images = len(image_groups)
    print(f"  Processing {total_images} images...")

    for idx, (img_path, annotations) in enumerate(image_groups.items()):
        if (idx + 1) % 10 == 0:
            print(f"    [{idx + 1}/{total_images}]")

        image = cv2.imread(img_path)
        if image is None:
            print(f"    Warning: Could not read {img_path}")
            continue

        h, w = image.shape[:2]
        positive_bboxes = []

        # Extract features for positive samples
        for (bbox, cls_name) in annotations:
            if cls_name not in CLASS_TO_IDX:
                continue

            features = extractor.extract_from_region(image, tuple(bbox), w, h)
            X_list.append(features)
            y_list.append(CLASS_TO_IDX[cls_name])
            positive_bboxes.append(tuple(bbox))

        # Generate & extract negative samples (3x positives for better balance)
        if include_negatives:
            neg_bboxes = generate_negative_samples(
                image, positive_bboxes, num_negatives=len(positive_bboxes) * 3
            )
            bg_idx = len(CLASSES)  # "background" class index
            for neg_bbox in neg_bboxes:
                features = extractor.extract_from_region(image, neg_bbox, w, h)
                X_list.append(features)
                y_list.append(bg_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution:")
    for i, cls_name in enumerate(CLASSES_WITH_BG):
        count = np.sum(y == i)
        if count > 0:
            print(f"    {cls_name:20s}: {count}")

    return X, y


# ── Training ────────────────────────────────────────────────────────

def _safe_split(X, y, test_size=0.2, random_state=42):
    """
    Smart train/test split that handles small datasets.
    Falls back to non-stratified or train-only when data is too small.
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    min_test = max(n_classes, 2)  # Need at least 1 per class for stratified
    min_samples_needed = int(min_test / test_size) + n_classes

    if n_samples >= min_samples_needed:
        # Enough data for stratified split
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)
    elif n_samples > n_classes * 2:
        # Small dataset: use non-stratified split
        print(f"  ⚠️  Small dataset ({n_samples} samples). Using non-stratified split.")
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)
    else:
        # Very small dataset: train on all data, use same data for eval
        print(f"  ⚠️  Very small dataset ({n_samples} samples). Training on all data (no holdout).")
        print(f"       Collect more data for reliable evaluation!")
        return X, X, y, y


def train_random_forest(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train a Random Forest classifier.

    Returns dict with: model, scaler, metrics
    """
    print("\n  Training Random Forest Classifier...")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Adaptive train/test split
    X_train, X_test, y_train, y_test = _safe_split(X, y, test_size=0.2)

    # Feature scaling (important for SVM, helps RF too)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Adapt min_samples_split to dataset size
    min_split = min(5, max(2, len(X_train) // 4))

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,            # Let trees grow fully — better for small datasets
        min_samples_split=min_split,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",   # Handle imbalanced classes
        bootstrap=True,
        oob_score=True,            # Out-of-bag score (free validation)
        n_jobs=-1,                 # Use all CPU cores
        random_state=42,
        verbose=1,
    )

    start = time.time()
    rf.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)

    print(f"\n  ✅ Training complete in {train_time:.1f}s")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASSES_WITH_BG[i] for i in np.unique(y_test)],
        zero_division=0,
    ))

    # Cross-validation (adapt folds to dataset size)
    n_folds = min(5, min(np.bincount(y_train)))  # Can't have more folds than smallest class
    n_folds = max(2, n_folds)  # At least 2-fold
    cv_mean = 0.0
    if len(X_train) >= n_folds * 2:
        print(f"  Running {n_folds}-fold cross-validation...")
        cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=n_folds, n_jobs=-1)
        print(f"  CV scores: {cv_scores}")
        print(f"  CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        cv_mean = cv_scores.mean()
    else:
        print("  ⚠️  Too few samples for cross-validation. Skipping.")

    # Feature importance (top 20)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-20:][::-1]
    print(f"\n  Top 20 most important features:")
    for i, idx in enumerate(top_indices):
        print(f"    {i + 1:2d}. Feature[{idx}] = {importances[idx]:.4f}")

    return {
        "model": rf,
        "scaler": scaler,
        "accuracy": accuracy,
        "cv_mean": cv_mean,
        "train_time": train_time,
        "classes": CLASSES_WITH_BG,
    }


def train_svm(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train an SVM classifier (alternative to Random Forest).
    """
    print("\n  Training SVM Classifier...")

    X_train, X_test, y_train, y_test = _safe_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,          # Enable probability estimates
        verbose=True,
    )

    start = time.time()
    svm.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = svm.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)

    print(f"\n  ✅ Training complete in {train_time:.1f}s")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASSES_WITH_BG[i] for i in np.unique(y_test)],
        zero_division=0,
    ))

    return {
        "model": svm,
        "scaler": scaler,
        "accuracy": accuracy,
        "train_time": train_time,
        "classes": CLASSES_WITH_BG,
    }


# ── Save / Load ────────────────────────────────────────────────────

def save_model(result: dict, model_type: str = "rf"):
    """Save trained model, scaler, and metadata."""
    model_path = MODEL_DIR / f"ui_detector_{model_type}.pkl"

    save_data = {
        "model": result["model"],
        "scaler": result["scaler"],
        "classes": result["classes"],
        "accuracy": result["accuracy"],
        "model_type": model_type,
    }

    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n  💾 Model saved: {model_path} ({size_mb:.1f} MB)")
    print(f"     Accuracy: {result['accuracy']:.4f}")
    return model_path


def load_model(model_type: str = "rf") -> dict:
    """Load a trained model."""
    model_path = MODEL_DIR / f"ui_detector_{model_type}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Loaded model: {model_path} (accuracy: {data['accuracy']:.4f})")
    return data


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train UI Element Classifier")
    parser.add_argument("--model", choices=["rf", "svm"], default="rf",
                        help="Model type: rf (Random Forest) or svm (SVM)")
    parser.add_argument("--augment", action="store_true",
                        help="Include augmented data in training")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  UI Element Classifier Training")
    print("=" * 60)

    # Load dataset
    samples = load_dataset(include_augmented=args.augment)

    if len(samples) == 0:
        print("\n  ❌ No labeled data found!")
        print("  Run 'python ml_model/collect_data.py' first to collect & label screenshots.")
        print("  Then run this script again.\n")
        sys.exit(1)

    # Build feature matrix
    X, y = build_feature_matrix(samples, include_negatives=True)

    if len(X) < 10:
        print("\n  ⚠️  Very few samples. Collect more data for better accuracy.")

    # Train
    if args.model == "rf":
        result = train_random_forest(X, y)
    else:
        result = train_svm(X, y)

    # Save
    save_model(result, model_type=args.model)

    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print("=" * 60 + "\n")
