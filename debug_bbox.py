"""Compare training search_bar bboxes vs live proposals"""
import json, glob, cv2, numpy as np
from collections import defaultdict

# Load training search bar bboxes
sb_bboxes = []
for f in sorted(glob.glob('ml_model/data/labels/*.json') + glob.glob('ml_model/data/augmented_labels/*.json')):
    data = json.load(open(f))
    for r in data:
        if r['class'] == 'search_bar':
            sb_bboxes.append(r['bbox'])

print(f"Training search_bar bboxes ({len(sb_bboxes)} total):")
widths = [b[2]-b[0] for b in sb_bboxes]
heights = [b[3]-b[1] for b in sb_bboxes]
y_centers = [(b[1]+b[3])/2 for b in sb_bboxes]
x_centers = [(b[0]+b[2])/2 for b in sb_bboxes]

print(f"  Width  range: {min(widths)} - {max(widths)} (mean {np.mean(widths):.0f})")
print(f"  Height range: {min(heights)} - {max(heights)} (mean {np.mean(heights):.0f})")
print(f"  Y-center range: {min(y_centers):.0f} - {max(y_centers):.0f}")
print(f"  X-center range: {min(x_centers):.0f} - {max(x_centers):.0f}")

# Show a few examples
print(f"\n  First 5 bboxes:")
for b in sb_bboxes[:5]:
    print(f"    ({b[0]},{b[1]},{b[2]},{b[3]})  size={b[2]-b[0]}x{b[3]-b[1]}")

# Now check live proposals in that region
from ml_model.detector import UIDetectorRF
import mss

with mss.mss() as sct:
    img = sct.grab(sct.monitors[1])
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

det = UIDetectorRF(model_type='rf', confidence_threshold=0.1)
proposals = det._generate_proposals(frame, max_proposals=300)

# Find proposals that overlap with typical search bar region
avg_y = np.mean(y_centers)
avg_x = np.mean(x_centers)
avg_w = np.mean(widths)
avg_h = np.mean(heights)
print(f"\nAverage training search_bar: center=({avg_x:.0f},{avg_y:.0f}) size={avg_w:.0f}x{avg_h:.0f}")

# Proposals close to search bar region
print(f"\nProposals near training search_bar position:")
count = 0
for p in proposals:
    px, py = (p[0]+p[2])/2, (p[1]+p[3])/2
    pw, ph = p[2]-p[0], p[3]-p[1]
    # Within 200px of average search bar center, similar aspect ratio
    if abs(py - avg_y) < 100 and abs(px - avg_x) < 500 and pw > 150:
        count += 1
        if count <= 10:
            print(f"  ({p[0]},{p[1]},{p[2]},{p[3]}) size={pw}x{ph}")
print(f"  Total matching: {count}")
