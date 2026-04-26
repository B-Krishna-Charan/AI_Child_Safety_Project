# AI Child Safety System
## AI-Driven Missing Child Detection with Web Dashboard

---

## What It Does

- **Web Dashboard** — Add missing persons with photos, view all records, filter by status
- **Face Recognition** — Matches faces from webcam against your registered missing persons database in real time
- **Live Feed** — SSE-powered terminal feed shows every frame scan result in the browser
- **Auto-Update** — When a match is detected, the person's status automatically flips to "Found"
- **REST API** — Full JSON API so the system can be extended or integrated with other services

---

## Project Structure

```
AI_Child_Safety_Project/
├── app.py                  ← Flask backend + face recognition engine
├── requirements.txt        ← Python dependencies
├── README.txt              ← This file
├── database.json           ← Auto-created: persists all person records
├── templates/
│   └── index.html          ← Web frontend (served by Flask)
├── missing_children/       ← Auto-populated: face images saved here on upload
└── uploads/                ← Temp folder for incoming uploads
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note on face_recognition:** This library requires `dlib` which needs cmake and a C++ compiler.
> On Ubuntu/Debian:  `sudo apt-get install build-essential cmake`
> On macOS:          `brew install cmake`
> On Windows:        Install Visual Studio Build Tools, then run the pip install.

### 2. Run the server

```bash
python app.py
```

### 3. Open the dashboard

Open your browser and go to:

```
http://localhost:5000
```

---

## How to Use

### Adding a Missing Person
1. Click "Photo Upload" and select a clear face photo of the child
2. Fill in the details (name is required; all other fields are optional)
3. Click **+ Add to Database**

The photo is saved to `missing_children/` and the face is encoded into the recognition model.

### Starting Live Tracking
1. Click **▶ Start Live Tracking** in the top-right of the database panel
2. The system opens your webcam and begins scanning faces in real time
3. Every frame result appears in the live terminal feed at the bottom
4. When a match is found (confidence ≥ 50%), an alert pops up and the record is marked **Found**
5. Click **■ Stop Tracking** to close the camera

### Marking Someone as Found Manually
Click the **✓ Found** button on any person card.

---

## API Reference

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | `/api/persons`         | List all person records            |
| POST   | `/api/persons`         | Add a new missing person (JSON)    |
| PATCH  | `/api/persons/<id>`    | Update a record (e.g. set status)  |
| DELETE | `/api/persons/<id>`    | Remove a record                    |
| GET    | `/api/stats`           | Summary stats (total/missing/found)|
| POST   | `/api/tracking/start`  | Start the face recognition loop    |
| POST   | `/api/tracking/stop`   | Stop the face recognition loop     |
| GET    | `/api/tracking/status` | Check if tracking is running       |
| GET    | `/api/events`          | SSE stream for live feed updates   |

### POST /api/persons — Body Schema
```json
{
  "name":     "Child Name",
  "age":      "10",
  "gender":   "Male",
  "date":     "2025-04-20",
  "location": "Hyderabad",
  "contact":  "Guardian Name",
  "phone":    "+91 9876543210",
  "desc":     "Wearing blue shirt...",
  "photo":    "data:image/jpeg;base64,..."
}
```

---

## Simulation Mode

If a webcam is not available (or `face_recognition` is not installed), the system automatically falls back to **simulation mode** — the live feed shows realistic simulated scan output so you can still demo and use the full dashboard without hardware.

---

## Requirements

- Python 3.8+
- A webcam (for live tracking; simulation mode works without one)
- Modern web browser (Chrome, Firefox, Edge)
