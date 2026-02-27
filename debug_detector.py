"""Quick diagnostic: what does the RF model see on the current screen?"""
import cv2
import numpy as np
import mss
import pickle
from collections import Counter
from ml_model.detector import UIDetectorRF
from ml_model.feature_extractor import FeatureExtractor

# ── Load model info ─────────────────────────────────────────────
with open("models/ui_detector_rf.pkl", "rb") as f:
    data = pickle.load(f)

classes = data["classes"]
print("Model classes:", classes)
print("Accuracy:", data.get("accuracy"))
print()

# ── Capture screenshot ──────────────────────────────────────────
with mss.mss() as sct:
    img = sct.grab(sct.monitors[1])
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

h, w = frame.shape[:2]
print(f"Screenshot: {w}x{h}")

# ── Generate proposals ──────────────────────────────────────────
det = UIDetectorRF(model_type="rf", confidence_threshold=0.1)
det._load_model()

proposals = det._generate_proposals(frame, max_proposals=300)
print(f"Total proposals: {len(proposals)}")

# ── Classify all proposals ──────────────────────────────────────
fe = FeatureExtractor()
features = [fe.extract_from_region(frame, bbox, w, h) for bbox in proposals]
X = np.array(features, dtype=np.float32)
X_scaled = det._scaler.transform(X)

probs = det._model.predict_proba(X_scaled)
preds = np.argmax(probs, axis=1)
confs = np.max(probs, axis=1)

pred_labels = [classes[p] for p in preds]
print(f"\n── Prediction distribution (ALL {len(proposals)} proposals) ──")
for label, count in Counter(pred_labels).most_common():
    idxs = [i for i, l in enumerate(pred_labels) if l == label]
    c = confs[idxs]
    print(f"  {label:20s}: {count:4d}  (conf {c.min():.3f} - {c.max():.3f})")

# ── Specifically look at search_bar ─────────────────────────────
sb_idx = classes.index("search_bar") if "search_bar" in classes else -1
if sb_idx >= 0:
    sb_probs = probs[:, sb_idx]
    top10 = np.argsort(sb_probs)[-10:][::-1]
    print(f"\n── Top 10 proposals with highest search_bar probability ──")
    for i in top10:
        x1, y1, x2, y2 = proposals[i]
        print(f"  bbox=({x1:4d},{y1:4d},{x2:4d},{y2:4d})  "
              f"size={x2-x1:4d}x{y2-y1:3d}  "
              f"search_bar_prob={sb_probs[i]:.4f}  "
              f"predicted={pred_labels[i]}({confs[i]:.3f})")

# ── Check: do ANY proposals cover the actual YT search bar area? ──
# YouTube search bar is roughly centered, y~55-85, width~500-600
print(f"\n── Proposals in YouTube search bar region (y:40-100, centered) ──")
center_x = w // 2
for i, (x1, y1, x2, y2) in enumerate(proposals):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = x2 - x1
    if 40 <= cy <= 100 and abs(cx - center_x) < 400 and bw > 200:
        print(f"  [{i:3d}] bbox=({x1},{y1},{x2},{y2}) size={bw}x{y2-y1}  "
              f"predicted={pred_labels[i]}({confs[i]:.3f})  "
              f"search_bar_prob={probs[i, sb_idx]:.4f}")

# ── Save annotated screenshot for visual check ─────────────────
vis = frame.copy()
# Draw all proposals that have search_bar_prob > 0.05
for i in range(len(proposals)):
    if sb_probs[i] > 0.05:
        x1, y1, x2, y2 = proposals[i]
        color = (0, int(255 * sb_probs[i]), 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"sb:{sb_probs[i]:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("debug_detections.png", vis)
print(f"\nSaved debug_detections.png — check it visually!")
