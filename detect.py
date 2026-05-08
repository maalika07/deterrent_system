"""
detect.py  v7  —  YOLO (detect) + MOG2 (motion) + ResNet50 (classify)
=======================================================================

WHY EVERYTHING WAS PREDICTED AS ELEPHANT:
  ResNet is trained on animal crops. When fed a background/non-animal image,
  it CANNOT say "nothing here" unless:
    1. The "no animal" class in the model is well-trained, AND
    2. Confidence threshold is HIGH enough to reject weak predictions.
  
  Elephant was winning because:
    - Background textures (grey walls, floors, soil) match elephant texture
    - CONF_THRESH was 0.55 — too low — elephant at 58% = false detection
    - MOG2 was sending shadow/noise blobs to ResNet
    - Full-frame scan was sending empty backgrounds to ResNet

FIXES:
  1. CONF_THRESH raised to 0.80 (YOLO crop) — only accept strong predictions
  2. MOG2_CONF raised to 0.85 — MOG2 blobs are noisier, need higher bar
  3. GAP_THRESH raised to 0.25 — top-1 must be clearly better than top-2
  4. ENT_THRESH lowered to 1.50 — model must be very certain
  5. NO full-frame scan in camera mode — background never touches ResNet
  6. MOG2 blob minimum size increased — filter shadows and small noise
  7. YOLO minimum box size increased — filter tiny false detections

ROLES (strict separation):
  YOLO   → bounding box ONLY. Class label completely IGNORED.
  MOG2   → moving blob ONLY. Gives crop coordinates for non-COCO species.
  ResNet → THE ONLY classifier. Names the animal including "no animal".
"""

# ── MKL-DNN FIX (must be before any torch import) ────────────────────────────
import os
os.environ["PYTORCH_JIT"]           = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"

import torch
torch.backends.mkldnn.enabled = False

import cv2
import torch.nn as nn
import argparse
import math
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESNET_PATH = "animal_classifier_resnet50.pth"
YOLO_MODEL  = "yolov8n.pt"

# ── CRITICAL THRESHOLDS ───────────────────────────────────────────────────────
# These were too low before — causing elephant false positives
# Rule: if in doubt, reject. Better to miss an animal than false trigger.

YOLO_CONF  = 0.20     # YOLO detection confidence (low — ResNet filters)
CONF_THRESH = 0.80    # ResNet confidence for YOLO crops  (was 0.55-0.60 → WRONG)
ENT_THRESH  = 1.50    # max entropy for YOLO crops        (was 1.80-2.20 → WRONG)
GAP_THRESH  = 0.25    # min top1-top2 gap for YOLO crops  (was 0.10-0.15 → WRONG)

# MOG2 crops are noisier — need even higher confidence
MOG2_CONF  = 0.85     # was 0.55-0.62 → caused elephant false positives
MOG2_ENT   = 1.20     # was 1.80-2.00 → too relaxed
MOG2_GAP   = 0.30     # was 0.10-0.13 → too relaxed

# Image/folder scan — strict to avoid background false positives
SCAN_CONF  = 0.82
SCAN_ENT   = 1.30
SCAN_GAP   = 0.28

MIN_BOX_W  = 60       # was 40 — larger minimum to filter noise
MIN_BOX_H  = 60
MOG2_BLOB_MIN = 4000  # was 1800-2500 — larger to filter shadows/noise
MOG2_BLOB_MAX = 160000
MOG2_PAD   = 25
MOG2_WARMUP = 10      # wait 10 frames before using MOG2

CONFIRM_FRAMES = 5    # consecutive frames before printing confirmation
MOTION_THRESH  = 30

CLASSES = [
    'Peacock', 'bonnet_macaque', 'chital', 'elephant',
    'no animal', 'pig', 'porcupine', 'street_dogs', 'wild_boar'
]
NO_ANIMAL_IDX = CLASSES.index('no animal')  # 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# COCO class IDs — only used to filter non-animal YOLO detections
# YOLO's label name is NEVER used for classification
ANIMAL_COCO_IDS = {14,15,16,17,18,19,20,21,22,23}


# ── LOAD MODELS ───────────────────────────────────────────────────────────────
def load_models():
    yolo = YOLO(YOLO_MODEL)
    yolo.to("cpu")  # CPU + MKL-DNN disabled = no primitive error
    print(f"[YOLO]     {YOLO_MODEL} — detection only (label IGNORED)")

    net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    net.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    net = net.to(DEVICE).eval()
    print(f"[ResNet50] {len(CLASSES)} classes on {DEVICE}")
    print(f"[ResNet50] Thresholds: conf>={CONF_THRESH} ent<={ENT_THRESH} gap>={GAP_THRESH}")
    return yolo, net


# ── PREPROCESS ────────────────────────────────────────────────────────────────
_prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── RESNET INFERENCE ──────────────────────────────────────────────────────────
def resnet_classify(net, patch):
    """
    Run ResNet50 on a patch.
    Returns: { name, idx, conf, ent, gap }
    
    IMPORTANT: "no animal" is a valid output.
    If ResNet is uncertain (low conf, high entropy, small gap) → "no animal".
    """
    if patch is None or patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
        return {"name":"no animal","idx":NO_ANIMAL_IDX,
                "conf":0.0,"ent":9.9,"gap":0.0}
    try:
        with torch.no_grad():
            probs = torch.nn.functional.softmax(
                net(_prep(patch).unsqueeze(0).to(DEVICE)), dim=1)[0]
        sp, si = probs.sort(descending=True)
        return {
            "name": CLASSES[si[0].item()],
            "idx":  si[0].item(),
            "conf": sp[0].item(),
            "ent":  -(probs * (probs + 1e-9).log()).sum().item(),
            "gap":  sp[0].item() - sp[1].item(),
        }
    except Exception as e:
        print(f"[ResNet ERROR] {e}")
        return {"name":"no animal","idx":NO_ANIMAL_IDX,
                "conf":0.0,"ent":9.9,"gap":0.0}


def passes_gate(pred, conf=CONF_THRESH, ent=ENT_THRESH, gap=GAP_THRESH):
    """
    All 4 gates must pass. If any fails → reject as "no animal".
    
    Gate 1: ResNet top prediction is NOT "no animal"
    Gate 2: Confidence >= threshold (rejects weak/uncertain predictions)
    Gate 3: Entropy <= threshold   (rejects confused/uncertain model state)
    Gate 4: Gap >= threshold       (top-1 must clearly beat top-2)
    """
    return (
        pred["idx"]  != NO_ANIMAL_IDX
        and pred["conf"] >= conf
        and pred["ent"]  <= ent
        and pred["gap"]  >= gap
    )


# ── MOG2 ─────────────────────────────────────────────────────────────────────
class MOG2Detector:
    """
    Background subtractor — finds moving objects regardless of species.
    Used for non-COCO species (macaque, chital, pig, porcupine, wild_boar).
    Strict blob size filter to remove shadows and camera noise.
    """
    def __init__(self):
        self._sub    = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._count  = 0

    def get_boxes(self, frame):
        self._count += 1
        if self._count < MOG2_WARMUP:
            return []

        fh, fw = frame.shape[:2]
        mask   = self._sub.apply(frame)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  self._kernel, iterations=3)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   self._kernel, iterations=2)
        mask   = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MOG2_BLOB_MIN <= area <= MOG2_BLOB_MAX):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            x1 = max(0,  bx - MOG2_PAD)
            y1 = max(0,  by - MOG2_PAD)
            x2 = min(fw, bx + bw + MOG2_PAD)
            y2 = min(fh, by + bh + MOG2_PAD)
            boxes.append((x1, y1, x2, y2))
        return _merge_boxes(boxes)


def _merge_boxes(boxes):
    if not boxes: return []
    merged = True
    while merged:
        merged = False
        out    = []
        used   = [False] * len(boxes)
        for i, (ax1,ay1,ax2,ay2) in enumerate(boxes):
            if used[i]: continue
            for j, (bx1,by1,bx2,by2) in enumerate(boxes):
                if i == j or used[j]: continue
                if max(ax1,bx1) < min(ax2,bx2) and max(ay1,by1) < min(ay2,by2):
                    ax1=min(ax1,bx1); ay1=min(ay1,by1)
                    ax2=max(ax2,bx2); ay2=max(ay2,by2)
                    used[j] = True; merged = True
            out.append((ax1,ay1,ax2,ay2)); used[i] = True
        boxes = out
    return boxes


# ── MOTION TRACKER ────────────────────────────────────────────────────────────
class MotionTracker:
    def __init__(self):
        self._prev = {}

    def update(self, slot_id, x1, y1, x2, y2):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        prev   = self._prev.get(slot_id)
        moving = (prev is not None and
                  math.sqrt((cx-prev[0])**2+(cy-prev[1])**2) > MOTION_THRESH)
        self._prev[slot_id] = (cx, cy)
        return moving


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def process_frame(yolo, net, tracker, frame,
                  mog2: MOG2Detector = None,
                  draw: bool = True,
                  allow_scan: bool = False):
    """
    CAMERA mode (allow_scan=False):
      Stage 1 — YOLO boxes → ResNet (high confidence gates)
      Stage 2 — MOG2 blobs → ResNet (very high confidence gates)
      Empty frame → [] → "no animal"  (background NEVER touches ResNet)

    IMAGE mode (allow_scan=True):
      Stage 1 — YOLO boxes → ResNet
      Stage 2 — 9-crop scan → ResNet with strict gates

    Returns list of dicts: {name, conf, ent, moving, x1,y1,x2,y2, source}
    Empty list = "no animal"
    """
    fh, fw = frame.shape[:2]

    # ── YOLO detection ────────────────────────────────────────────────────────
    try:
        yolo_res = yolo(frame, verbose=False)[0]
    except Exception as e:
        print(f"[YOLO ERROR] {e}")
        yolo_res = None

    detections = []

    # ── Stage 1: YOLO box → ResNet ────────────────────────────────────────────
    if yolo_res is not None:
        candidates = []
        for i, box in enumerate(yolo_res.boxes):
            cls_id    = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            if cls_id not in ANIMAL_COCO_IDS or yolo_conf < YOLO_CONF:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1=max(0,x1); y1=max(0,y1)
            x2=min(fw,x2); y2=min(fh,y2)
            w = x2-x1; h = y2-y1
            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue
            candidates.append((w*h, i, x1,y1,x2,y2))

        # Largest box first — most likely the main animal
        for _, slot_id, x1,y1,x2,y2 in sorted(candidates, reverse=True)[:5]:
            crop   = frame[y1:y2, x1:x2]
            pred   = resnet_classify(net, crop)
            passed = passes_gate(pred)  # uses CONF_THRESH=0.80
            moving = tracker.update(slot_id, x1,y1,x2,y2)

            if draw:
                color = (0,255,0) if passed else (60,60,60)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                if passed:
                    lbl = f"{pred['name']}  {pred['conf']:.0%}" + (" ⚡" if moving else "")
                else:
                    lbl = f"? {pred['conf']:.0%}"
                cv2.putText(frame, lbl, (x1, max(y1-8,14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if passed:
                detections.append(dict(
                    name=pred["name"], conf=pred["conf"], ent=pred["ent"],
                    moving=moving, x1=x1,y1=y1,x2=x2,y2=y2,
                    source="yolo"))

    if detections:
        return detections

    # ── Stage 2a: Camera — MOG2 motion blobs → ResNet ────────────────────────
    # Only for MOVING objects. High gate (0.85) prevents background = elephant.
    # Nothing moves → [] → "no animal". Background never touches ResNet.
    if mog2 is not None and not allow_scan:
        for mx1,my1,mx2,my2 in mog2.get_boxes(frame)[:3]:
            bw = mx2-mx1; bh = my2-my1
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue
            crop   = frame[my1:my2, mx1:mx2]
            pred   = resnet_classify(net, crop)
            passed = passes_gate(pred, conf=MOG2_CONF, ent=MOG2_ENT, gap=MOG2_GAP)

            if draw:
                color = (255,165,0) if passed else (60,60,60)
                cv2.rectangle(frame,(mx1,my1),(mx2,my2),color,2)
                if passed:
                    cv2.putText(frame,
                                f"{pred['name']}  {pred['conf']:.0%}  ⚡MOG2",
                                (mx1, max(my1-8,14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if passed:
                detections.append(dict(
                    name=pred["name"], conf=pred["conf"], ent=pred["ent"],
                    moving=True, x1=mx1,y1=my1,x2=mx2,y2=my2,
                    source="mog2"))

        return detections  # empty = "no animal" — correct behaviour

    # ── Stage 2b: Image/folder — 9 crop scan → ResNet ────────────────────────
    if allow_scan:
        best = _best_scan(net, frame)
        if best:
            pred,(x1,y1,x2,y2),lbl = best
            if draw:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),3)
                cv2.putText(frame,
                            f"{pred['name']}  {pred['conf']:.0%}  [{lbl}]",
                            (x1+4, max(y1+22,22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,220,0),2)
            detections.append(dict(
                name=pred["name"], conf=pred["conf"], ent=pred["ent"],
                moving=False, x1=x1,y1=y1,x2=x2,y2=y2,
                source=f"scan:{lbl}"))

    return detections


# ── IMAGE SCAN ────────────────────────────────────────────────────────────────
def _scan_crops(frame):
    h,w   = frame.shape[:2]
    h2,w2 = h//2, w//2
    cx1,cx2 = w//4, 3*w//4
    cy1,cy2 = h//4, 3*h//4
    return [
        (frame,                  "full",   0,   0,   w,   h),
        (frame[cy1:cy2,cx1:cx2],"center", cx1, cy1, cx2, cy2),
        (frame[0:h2,   0:w2],   "tl",     0,   0,   w2,  h2),
        (frame[0:h2,   w2:w],   "tr",     w2,  0,   w,   h2),
        (frame[h2:h,   0:w2],   "bl",     0,   h2,  w2,  h),
        (frame[h2:h,   w2:w],   "br",     w2,  h2,  w,   h),
    ]

def _best_scan(net, frame):
    best_pred = best_box = best_lbl = None
    best_c    = -1.0
    for crop,lbl,x1,y1,x2,y2 in _scan_crops(frame):
        if crop.size == 0: continue
        pred = resnet_classify(net, crop)
        if passes_gate(pred, conf=SCAN_CONF, ent=SCAN_ENT, gap=SCAN_GAP):
            if pred["conf"] > best_c:
                best_pred=pred; best_c=pred["conf"]
                best_box=(x1,y1,x2,y2); best_lbl=lbl
    return (best_pred,best_box,best_lbl) if best_pred else None


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA MODE
# ══════════════════════════════════════════════════════════════════════════════
def run_camera_mode(yolo, net):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    if not cap.isOpened(): print("[ERROR] No camera."); return

    print(f"[CAMERA] conf>={CONF_THRESH} ent<={ENT_THRESH} gap>={GAP_THRESH}")
    print("[INFO]   Empty frame = 'no animal' (background never classified)")

    tracker    = MotionTracker()
    mog2       = MOG2Detector()
    consec     = {}
    confirmed  = {}
    prev_print = set()

    while True:
        ret, frame = cap.read()
        if not ret: break

        dets         = process_frame(yolo, net, tracker, frame,
                                     mog2=mog2, draw=True, allow_scan=False)
        detected_now = {d["name"] for d in dets}

        for cls in list(consec.keys()):
            consec[cls] = consec[cls]+1 if cls in detected_now else 0
        for cls in detected_now:
            consec.setdefault(cls, 1)
        for cls in detected_now:
            if consec.get(cls,0) >= CONFIRM_FRAMES:
                confirmed[cls] = next(x for x in dets if x["name"]==cls)
        for cls in list(confirmed.keys()):
            if consec.get(cls,0) == 0:
                del confirmed[cls]

        cur = set(confirmed.keys())
        for cls in cur - prev_print:
            d = confirmed[cls]
            print(f"[CONFIRM]  {cls:<18} conf={d['conf']:.1%}  "
                  f"src={d['source']}  moving={d['moving']}")
        for cls in prev_print - cur:
            print(f"[GONE]     {cls}")
        if not cur and prev_print:
            print("[--]       no animal")
        prev_print = cur

        fh2,fw2 = frame.shape[:2]
        cv2.rectangle(frame,(0,0),(fw2,50),(0,0,0),-1)
        if confirmed:
            lbl = "  |  ".join(
                f"{c}  {confirmed[c]['conf']:.0%}"
                + (" ⚡" if confirmed[c]['moving'] else "")
                for c in confirmed)
            cv2.putText(frame, lbl, (10,34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No animal detected",
                        (10,34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120,120,120), 2)

        cv2.imshow("AI Scarecrow — YOLO+MOG2+ResNet50", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
def run_image_mode(yolo, net, image_path):
    image = cv2.imread(image_path)
    if image is None: print(f"[ERROR] Cannot read: {image_path}"); return

    tracker = MotionTracker()
    dets    = process_frame(yolo, net, tracker, image,
                            mog2=None, draw=True, allow_scan=True)
    if not dets:
        print(f"[RESULT]  No animal  ({os.path.basename(image_path)})")
    else:
        for d in dets:
            print(f"[RESULT]  {d['name']:<18} conf={d['conf']:.1%}  "
                  f"ent={d['ent']:.2f}  gap={d['conf']-d['ent']:.2f}  src={d['source']}")

    dh,dw = image.shape[:2]
    if max(dh,dw) > 900:
        s=900/max(dh,dw); image=cv2.resize(image,(int(dw*s),int(dh*s)))
    cv2.imshow(f"Result — {os.path.basename(image_path)}", image)
    print("[INFO]  Press any key to close")
    cv2.waitKey(0); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER MODE
# ══════════════════════════════════════════════════════════════════════════════
def run_folder_mode(yolo, net, folder_path):
    files = sorted([f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
    if not files: print(f"[ERROR] No images in: {folder_path}"); return

    print(f"[FOLDER] {len(files)} images — any key=next  q=quit")
    tracker = MotionTracker()

    for fname in files:
        path  = os.path.join(folder_path, fname)
        image = cv2.imread(path)
        if image is None: continue

        dets = process_frame(yolo, net, tracker, image,
                             mog2=None, draw=True, allow_scan=True)
        if not dets:
            print(f"  {fname:<40}  →  no animal")
        else:
            for d in dets:
                print(f"  {fname:<40}  →  {d['name']}  "
                      f"{d['conf']:.1%}  src={d['source']}")

        dh,dw = image.shape[:2]
        if max(dh,dw) > 900:
            s=900/max(dh,dw); image=cv2.resize(image,(int(dw*s),int(dh*s)))
        cv2.imshow("Folder Mode", image)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",        type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    args = parser.parse_args()
    yolo, net = load_models()
    if args.image:           run_image_mode(yolo, net, args.image)
    elif args.image_folder:  run_folder_mode(yolo, net, args.image_folder)
    else:                    run_camera_mode(yolo, net)


if __name__ == "__main__":
    main()