# yolo_streamlit.py
import os, io, time, uuid
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL_NAME = "yolov8n.pt"        # small, downloads automatically
PARQUET_FILE = "detected_violations.parquet"
CONF_THR = 0.35                 # detection confidence threshold
OUT_DIR = "yolo_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- UTIL ----------
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_NAME)

model = load_model()

def box_to_int(xyxy):
    return [int(round(v)) for v in xyxy]

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def area(box):
    x1,y1,x2,y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def iou_area(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    return interW * interH

def sample_color(img_pil, bbox):
    w,h = img_pil.size
    x1,y1,x2,y2 = [int(round(v)) for v in bbox]
    cx = min(max(int((x1+x2)/2.0), 0), w-1)
    cy = min(max(int((y1+y2)/2.0), 0), h-1)
    return img_pil.convert("RGB").getpixel((cx,cy))

def draw_violations(pil_img, vio_boxes):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for vb in vio_boxes:
        x1,y1,x2,y2,label,conf = vb
        draw.rectangle(((x1,y1),(x2,y2)), outline="red", width=3)
        text = f"{label} {conf:.2f}"
        draw.text((x1+4,y1+4), text, fill="yellow", font=font)
    return pil_img

def append_parquet_safely(df_new, path):
    try:
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df_concat = pd.concat([df_old, df_new], ignore_index=True)
            tmp_path = path + ".tmp"
            df_concat.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        else:
            tmp_path = path + ".tmp"
            df_new.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        return True, ""
    except Exception as e:
        return False, str(e)

# ---------- MAPPING RULES ----------
def map_detections_to_violations(detections, img_pil, location_input):
    # separate by labels
    persons = [d for d in detections if d["label"].lower() == "person"]
    helmets = [d for d in detections if d["label"].lower() == "helmet"]
    motorcycles = [d for d in detections if d["label"].lower() in ("motorbike","motorcycle","scooter","moped")]
    cars = [d for d in detections if d["label"].lower() in ("car","bus","truck","van")]
    traffic_lights = [d for d in detections if "traffic" in d["label"].lower()]

    violations = []
    annotated = []

    # --- No Helmet: only consider persons that are riding a motorcycle
    for p in persons:
        pbox = p["box"]
        pcenter = center(pbox)
        # check if this person is inside any motorcycle box OR very close to its bottom area (rider)
        riding = False
        for m in motorcycles:
            mbox = m["box"]
            if (pcenter[0] >= mbox[0] and pcenter[0] <= mbox[2] and
                pcenter[1] >= mbox[1] and pcenter[1] <= mbox[3]):
                riding = True
                related_mbox = mbox
                break
        if not riding:
            continue  # person not riding => skip helmet check

        # now check if any helmet overlaps the person's head area
        has_helmet = False
        parea = area(pbox)
        for h in helmets:
            a = iou_area(pbox, h["box"])
            if a > 0 and (a / max(1, parea)) > 0.02:
                has_helmet = True
                break
        if not has_helmet and p["conf"] >= 0.35:
            violations.append({
                "Violation_ID": str(uuid.uuid4())[:8],
                "Timestamp": datetime.utcnow().isoformat(),
                "Location": location_input,
                "Violation_Type": "No Helmet",
                "Vehicle_Type": "Motorbike",
                "Severity": 3,
                "Source": "yolo",
                "Label": "no_helmet",
                "Confidence": float(p["conf"])
            })
            annotated.append((*box_to_int(pbox), "No Helmet", float(p["conf"])))

    # --- Triple riding: person centers inside a single motorcycle bbox >=3
    for m in motorcycles:
        mbox = m["box"]
        cnt = 0
        for p in persons:
            cx,cy = center(p["box"])
            if (cx >= mbox[0] and cx <= mbox[2] and cy >= mbox[1] and cy <= mbox[3]):
                cnt += 1
        if cnt >= 3 and m["conf"] >= 0.3:
            violations.append({
                "Violation_ID": str(uuid.uuid4())[:8],
                "Timestamp": datetime.utcnow().isoformat(),
                "Location": location_input,
                "Violation_Type": "Triple Riding",
                "Vehicle_Type": "Motorbike",
                "Severity": 4,
                "Source": "yolo",
                "Label": "triple_riding",
                "Confidence": float(m["conf"])
            })
            annotated.append((*box_to_int(m["box"]), "Triple Riding", float(m["conf"])))

    # --- Red light jumping: sample traffic light color + vehicle position
    red_light = None
    for tl in traffic_lights:
        pix = sample_color(img_pil, tl["box"])
        R,G,B = pix
        if (R > 120) and (R > (G + 30)) and (R > (B + 30)):  # simple red test
            red_light = tl
            break
    if red_light:
        tlbox = red_light["box"]
        tl_y = (tlbox[1] + tlbox[3]) / 2.0
        annotated.append((*box_to_int(tlbox), "Traffic Light (Red)", float(red_light["conf"])))
        candidates = motorcycles + cars
        for v in candidates:
            vbox = v["box"]
            v_bottom_y = vbox[3]
            if v_bottom_y > tl_y and v["conf"] >= 0.25:
                violations.append({
                    "Violation_ID": str(uuid.uuid4())[:8],
                    "Timestamp": datetime.utcnow().isoformat(),
                    "Location": location_input,
                    "Violation_Type": "Red Light Jumping",
                    "Vehicle_Type": v["label"].capitalize(),
                    "Severity": 5,
                    "Source": "yolo",
                    "Label": "red_light_jump",
                    "Confidence": float(v["conf"])
                })
                annotated.append((*box_to_int(vbox), "Red Light Jump", float(v["conf"])))

    # Return both DataFrame rows and annotated boxes
    return violations, annotated

# ---------- STREAMLIT UI ----------
st.title("YOLO-based Violation Detector")
st.write("Upload an image (.jpg/.png).")

if "location_input" not in st.session_state:
    st.session_state["location_input"] = "Intersection_demo"

location_input = st.text_input("Location (optional)", value=st.session_state["location_input"])
st.session_state["location_input"] = location_input

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded is not None:
    img_bytes = uploaded.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    if st.button("Detect"):
        t0 = time.time()
        results = model.predict(source=np.array(image), conf=CONF_THR, verbose=False)
        res = results[0]
        detections = []
        for det in res.boxes:
            xyxy = det.xyxy[0].cpu().numpy()
            conf = float(det.conf.cpu().numpy())
            cls = int(det.cls.cpu().numpy())
            name = model.names.get(cls, str(cls))
            detections.append({"label": name, "conf": conf, "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]})

        violations, ann_boxes = map_detections_to_violations(detections, image.copy(), location_input)

        # Draw annotated image (only violation boxes)
        out_img = image.copy()
        out_img = draw_violations(out_img, ann_boxes)
        st.image(out_img, caption="Violations (annotated)", use_column_width=True)

        # Save violations to parquet with required schema columns
        if violations:
            df_new = pd.DataFrame(violations)
            # keep only the canonical columns (same as simulated)
            keep_cols = ["Violation_ID","Timestamp","Location","Violation_Type","Vehicle_Type","Severity"]
            # ensure columns present
            for c in keep_cols:
                if c not in df_new.columns:
                    df_new[c] = None
            df_new = df_new[keep_cols]
            ok, err = append_parquet_safely(df_new, PARQUET_FILE)
            if ok:
                st.success(f"{len(df_new)} violations saved to {PARQUET_FILE}")
            else:
                st.error(f"Failed to save parquet: {err}")
        else:
            st.info("No violations detected by current rules.")

        st.write(f"Detection time: {time.time() - t0:.2f} sec")