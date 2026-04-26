"""
AI Child Safety System - Flask Backend
Connects the web frontend to the face recognition engine.

FIXES APPLIED:
  1. Single shared camera capture thread → eliminates dual-capture lag
  2. Model loads in parallel with camera warmup → faster tracking start
  3. CAP_PROP_BUFFERSIZE=1 → eliminates stale-frame buffer lag
  4. SSE throttled to ~1 msg/sec → less browser flooding
  5. MJPEG stream reads shared annotated_frame, no second VideoCapture
"""

import os
import json
import time
import base64
import threading
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER   = "uploads"
MISSING_FOLDER  = "missing_children"
DB_FILE         = "database.json"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "gif", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER,   exist_ok=True)
os.makedirs(MISSING_FOLDER,  exist_ok=True)

tracking_active  = False
tracking_thread  = None
capture_thread   = None
sse_clients      = []
sse_lock         = threading.Lock()

# FIX: shared frames between threads
latest_frame    = None
frame_lock      = threading.Lock()
annotated_frame = None
annotated_lock  = threading.Lock()

known_encodings = []
known_names     = []
model_ready     = False   # FIX: tracks whether model has finished loading

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def broadcast(event_type, payload):
    msg = f"data: {json.dumps({'type': event_type, 'payload': payload})}\n\n"
    with sse_lock:
        for q in sse_clients:
            q.append(msg)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/persons", methods=["GET"])
def get_persons():
    return jsonify(load_db())

@app.route("/api/persons", methods=["POST"])
def add_person():
    data = request.get_json(force=True)
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    persons   = load_db()
    person_id = str(uuid.uuid4())
    photo_url = None
    photo_b64 = data.get("photo")

    if photo_b64 and photo_b64.startswith("data:image"):
        try:
            header, encoded = photo_b64.split(",", 1)
            ext   = header.split("/")[1].split(";")[0]
            fname = f"{person_id}.{ext}"
            fpath = os.path.join(MISSING_FOLDER, fname)
            with open(fpath, "wb") as f:
                f.write(base64.b64decode(encoded))
            photo_url = f"/missing_children/{fname}"
        except Exception as e:
            print(f"Photo save error: {e}")

    person = {
        "id":       person_id,
        "name":     data.get("name"),
        "age":      data.get("age", "?"),
        "gender":   data.get("gender", "Unknown"),
        "date":     data.get("date", datetime.today().strftime("%Y-%m-%d")),
        "location": data.get("location", "Unknown"),
        "contact":  data.get("contact", "—"),
        "phone":    data.get("phone", "—"),
        "desc":     data.get("desc", ""),
        "photo":    photo_url,
        "status":   "missing",
        "created":  datetime.now().isoformat(),
    }
    persons.insert(0, person)
    save_db(persons)

    if tracking_active:
        threading.Thread(target=reload_recognition_model, daemon=True).start()
        broadcast("feed", {"text": f"New record: {person['name'].upper()} — model reloading", "cls": "alert"})

    return jsonify(person), 201

@app.route("/api/persons/<person_id>", methods=["PATCH"])
def update_person(person_id):
    persons = load_db()
    data    = request.get_json(force=True)
    for p in persons:
        if p["id"] == person_id:
            p.update({k: v for k, v in data.items() if k != "id"})
            save_db(persons)
            if data.get("status") == "found":
                broadcast("match", {"name": p["name"], "id": person_id})
            return jsonify(p)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/persons/<person_id>", methods=["DELETE"])
def delete_person(person_id):
    persons = load_db()
    person  = next((p for p in persons if p["id"] == person_id), None)
    if not person:
        return jsonify({"error": "Not found"}), 404
    if person.get("photo"):
        try:
            os.remove(person["photo"].lstrip("/"))
        except Exception:
            pass
    persons = [p for p in persons if p["id"] != person_id]
    save_db(persons)
    return jsonify({"deleted": True})

@app.route("/missing_children/<path:filename>")
def serve_photo(filename):
    from flask import send_from_directory
    return send_from_directory(MISSING_FOLDER, filename)

# ── Tracking endpoints ────────────────────────────────────────────

@app.route("/api/tracking/start", methods=["POST"])
def start_tracking():
    global tracking_active, tracking_thread, capture_thread, model_ready
    if tracking_active:
        return jsonify({"status": "already_running"})

    tracking_active = True
    model_ready     = False   # will be set True when reload finishes

    # Reload model so any newly added persons are included
    threading.Thread(target=reload_recognition_model, daemon=True).start()

    # Separate capture and recognition threads
    capture_thread  = threading.Thread(target=camera_capture_loop, daemon=True)
    tracking_thread = threading.Thread(target=run_tracking_loop,   daemon=True)
    capture_thread.start()
    tracking_thread.start()

    broadcast("feed", {"text": "Tracking system started — camera opening", "cls": ""})
    return jsonify({"status": "started"})

@app.route("/api/tracking/stop", methods=["POST"])
def stop_tracking():
    global tracking_active
    tracking_active = False
    broadcast("feed", {"text": "Tracking stopped — camera released", "cls": "alert"})
    return jsonify({"status": "stopped"})

@app.route("/api/tracking/status", methods=["GET"])
def tracking_status():
    return jsonify({"active": tracking_active})

@app.route("/api/tracking/frame")
def video_frame():
    """
    MJPEG stream. FIX: reads annotated_frame written by the recognition
    loop — never opens its own VideoCapture.
    """
    import cv2
    import numpy as np

    def generate():
        placeholder = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Connecting to camera...", (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 180, 255), 2)
        _, ph_buf = cv2.imencode(".jpg", placeholder)
        ph_bytes  = ph_buf.tobytes()

        while tracking_active:
            with annotated_lock:
                frame = annotated_frame

            if frame is None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + ph_bytes + b"\r\n"
                time.sleep(0.1)
                continue

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)  # ~30 fps

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})

@app.route("/api/events")
def sse_stream():
    client_queue = []
    with sse_lock:
        sse_clients.append(client_queue)

    def generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        try:
            while True:
                if client_queue:
                    yield client_queue.pop(0)
                else:
                    time.sleep(0.1)
        except GeneratorExit:
            with sse_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Face recognition model ────────────────────────────────────────

def reload_recognition_model():
    global known_encodings, known_names, model_ready
    try:
        import face_recognition
        import cv2
        persons   = load_db()
        encodings, names = [], []
        for p in persons:
            if p["status"] != "missing" or not p.get("photo"):
                continue
            path = p["photo"].lstrip("/")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if encs:
                encodings.append(encs[0])
                names.append(p["name"])
        known_encodings = encodings
        known_names     = names
        model_ready     = True   # FIX: signal model is ready
        broadcast("feed", {"text": f"Model ready — {len(names)} face(s) registered", "cls": "model_ready"})
        return True
    except ImportError:
        broadcast("feed", {"text": "face_recognition not installed — demo mode", "cls": "alert"})
        return False
    except Exception as e:
        broadcast("feed", {"text": f"Model error: {e}", "cls": "alert"})
        return False

# ── FIX: Single dedicated camera capture thread ───────────────────

def camera_capture_loop():
    """
    Owns the camera. Writes raw frames to latest_frame.
    Only one VideoCapture ever exists in the process.
    """
    global latest_frame, tracking_active
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        # FIX: buffer=1 means we always read the NEWEST frame, not stale ones
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            broadcast("feed", {"text": "Camera unavailable — simulation mode active", "cls": "alert"})
            return

        broadcast("feed", {"text": "Camera open — streaming at native fps", "cls": ""})

        while tracking_active:
            ok, frame = cap.read()
            if ok:
                with frame_lock:
                    latest_frame = frame
            # No sleep — grab frames as fast as possible to keep latest_frame fresh

        cap.release()
        with frame_lock:
            latest_frame = None

    except Exception as e:
        broadcast("feed", {"text": f"Camera error: {e}", "cls": "alert"})

# ── Recognition loop ─────────────────────────────────────────────

def run_tracking_loop():
    """
    Reads shared latest_frame, runs face recognition, writes annotated_frame.
    NOTE: does NOT start its own reload_recognition_model thread —
          startup pre-load and per-session reload are handled externally.
    """
    global tracking_active, annotated_frame

    try:
        import face_recognition
        import cv2
        import numpy as np

        frame_count   = 0
        last_sse_time = 0.0
        recog_interval = 3   # run recognition every N frames

        while tracking_active:
            with frame_lock:
                frame = latest_frame

            if frame is None:
                time.sleep(0.05)
                continue

            frame = frame.copy()
            frame_count += 1

            h, w = frame.shape[:2]
            ts = time.strftime("%H:%M:%S")
            cv2.rectangle(frame, (0, 0), (w, 30), (15, 60, 100), -1)
            cv2.putText(frame, f"AI CHILD SAFETY  |  {ts}  |  LIVE",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)

            # Show MODEL LOADING overlay until model is ready
            if not model_ready:
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, "MODEL LOADING — PLEASE WAIT...", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                with annotated_lock:
                    annotated_frame = frame
                time.sleep(0.03)
                continue   # skip recognition until model is ready

            # Snapshot encodings so we use a consistent list this frame
            cur_encodings = list(known_encodings)
            cur_names     = list(known_names)

            # Run recognition every recog_interval frames
            if frame_count % recog_interval == 0 and cur_encodings:
                small  = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_sm = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs   = face_recognition.face_locations(rgb_sm)
                encs   = face_recognition.face_encodings(rgb_sm, locs)

                now        = time.time()
                should_log = (now - last_sse_time) >= 1.0

                if not locs:
                    if should_log:
                        broadcast("feed", {"text": "Scanning — no face in frame", "cls": ""})
                        last_sse_time = now
                else:
                    if should_log:
                        broadcast("feed", {"text": f"{len(locs)} face(s) detected — comparing...", "cls": ""})
                        last_sse_time = now

                for enc, (t, r, b, l) in zip(encs, locs):
                    dists    = face_recognition.face_distance(cur_encodings, enc)
                    best_idx = int(np.argmin(dists))
                    best_d   = float(dists[best_idx])
                    conf     = round((1 - best_d) * 100, 1)
                    t2, r2, b2, l2 = t*4, r*4, b*4, l*4

                    if best_d < 0.5:
                        name  = cur_names[best_idx]
                        color = (0, 200, 80)
                        label = f"MATCH: {name.upper()}  {conf}%"
                        broadcast("match", {"name": name, "confidence": conf})
                        broadcast("feed", {"text": f"MATCH FOUND: {name.upper()} — {conf}%", "cls": "match"})
                        last_sse_time = now
                        persons = load_db()
                        for p in persons:
                            if p["name"] == name and p["status"] == "missing":
                                p["status"] = "found"
                        save_db(persons)
                    else:
                        color = (0, 140, 255)
                        label = f"Unknown  {conf}%"

                    cv2.rectangle(frame, (l2, t2), (r2, b2), color, 2)
                    cv2.rectangle(frame, (l2, b2 - 28), (r2, b2), color, cv2.FILLED)
                    cv2.putText(frame, label, (l2 + 5, b2 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "SCANNING...", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            with annotated_lock:
                annotated_frame = frame

            time.sleep(0.01)

        with annotated_lock:
            annotated_frame = None

    except ImportError:
        _run_simulation_mode()
    except Exception as e:
        broadcast("feed", {"text": f"Tracking error: {e} — simulation mode", "cls": "alert"})
        _run_simulation_mode()

def _run_simulation_mode():
    import random
    msgs = [
        "Frame analysed — no match detected",
        "Face detected — comparing against database",
        "Processing region of interest",
        "Scanning complete — below threshold",
        "Encoding comparison in progress",
        "No face in frame",
    ]
    persons = load_db()
    missing = [p["name"] for p in persons if p["status"] == "missing"]
    while tracking_active:
        msg = random.choice(msgs)
        if missing and random.random() < 0.15:
            name = random.choice(missing)
            conf = round(random.uniform(30, 48), 1)
            msg  = f"Encoding comparison: {name} — {conf}% (below threshold)"
        broadcast("feed", {"text": msg, "cls": ""})
        time.sleep(3.0)

# ── Stats ─────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    persons  = load_db()
    total    = len(persons)
    missing  = sum(1 for p in persons if p["status"] == "missing")
    found    = sum(1 for p in persons if p["status"] == "found")

    from datetime import date
    days_list = []
    for p in persons:
        if p["status"] == "missing" and p.get("date"):
            try:
                d = date.fromisoformat(p["date"])
                days_list.append((date.today() - d).days)
            except Exception:
                pass
    avg_days = round(sum(days_list) / len(days_list)) if days_list else None

    return jsonify({
        "total":    total,
        "missing":  missing,
        "found":    found,
        "avg_days": avg_days,
        "tracking": tracking_active,
    })

if __name__ == "__main__":
    print("\n🔵 AI Child Safety System")
    print("   Open http://localhost:5000 in your browser\n")
    # FIX: pre-load face encodings at startup so model is ready before user clicks Start
    threading.Thread(target=reload_recognition_model, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
