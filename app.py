from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, sqlite3, json, uuid, io, csv, hashlib, hmac, base64
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
from predict import run_prediction

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Ayushman Bharat Fraud Detection Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

for d in ["uploads", "outputs", "static"]:
    os.makedirs(d, exist_ok=True)

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = "fraud_detection.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'analyst',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_batch TEXT,
        claim_id TEXT,
        abha_id TEXT,
        patient_id TEXT,
        hospital_id TEXT,
        doctor_id TEXT,
        diagnosis_code TEXT,
        treatment_code TEXT,
        admission_date TEXT,
        discharge_date TEXT,
        claim_amount REAL,
        approved_amount REAL,
        location TEXT,
        risk_score REAL DEFAULT 0,
        risk_level TEXT DEFAULT 'Low',
        anomaly_score REAL DEFAULT 0,
        fraud_reasons TEXT DEFAULT '',
        status TEXT DEFAULT 'Pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS hospitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_id TEXT UNIQUE,
        name TEXT,
        location TEXT,
        total_claims INTEGER DEFAULT 0,
        flagged_claims INTEGER DEFAULT 0,
        avg_risk_score REAL DEFAULT 0,
        risk_level TEXT DEFAULT 'Low',
        total_amount REAL DEFAULT 0,
        suspicious_amount REAL DEFAULT 0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT,
        hospital_id TEXT,
        risk_score REAL,
        message TEXT,
        is_read INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS upload_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT UNIQUE,
        filename TEXT,
        total_claims INTEGER,
        high_risk INTEGER,
        medium_risk INTEGER,
        low_risk INTEGER,
        uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # Seed admin user
    pwd = _hash_password("admin123")
    c.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
              ("admin", pwd, "admin"))
    c.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
              ("analyst", _hash_password("analyst123"), "analyst"))
    conn.commit()
    # Migration: add abha_id column if not yet present (safe for existing DBs)
    try:
        conn.execute("ALTER TABLE claims ADD COLUMN abha_id TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.close()

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _make_token(username: str, role: str) -> str:
    payload = json.dumps({"username": username, "role": role,
                          "exp": (datetime.utcnow() + timedelta(hours=8)).isoformat()})
    return base64.b64encode(payload.encode()).decode()

def _verify_token(token: str):
    try:
        payload = json.loads(base64.b64decode(token).decode())
        if datetime.fromisoformat(payload["exp"]) < datetime.utcnow():
            return None
        return payload
    except:
        return None

init_db()

# ─── WebSocket Manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for conn in self.active:
            try:
                await conn.send_json(data)
            except:
                dead.append(conn)
        for d in dead:
            self.active.remove(d)

manager = ConnectionManager()

# ─── Fraud Explanation ────────────────────────────────────────────────────────
def generate_fraud_reasons(row, df):
    reasons = []
    if "Claim_Amount" in df.columns and "Hospital_ID" in df.columns:
        hosp_avg = df.groupby("Hospital_ID")["Claim_Amount"].mean()
        h_avg = hosp_avg.get(row.get("Hospital_ID", ""), 0)
        if h_avg > 0 and row.get("Claim_Amount", 0) > 2 * h_avg:
            reasons.append("Claim amount exceeds 2x hospital average")
    if "Patient_ID" in df.columns:
        dups = df["Patient_ID"].value_counts()
        if dups.get(row.get("Patient_ID", ""), 0) > 3:
            reasons.append("Duplicate patient ID detected (>3 claims)")
    if "Treatment_Code" in df.columns and "Doctor_ID" in df.columns:
        doc_tx = df.groupby("Doctor_ID")["Treatment_Code"].count()
        if doc_tx.get(row.get("Doctor_ID", ""), 0) > 10:
            reasons.append("Unusual treatment frequency for this doctor")
    if "Claim_Amount" in df.columns and "Approved_Amount" in df.columns:
        ca = row.get("Claim_Amount", 0)
        aa = row.get("Approved_Amount", 0)
        if aa > 0 and ca > 0 and (ca - aa) / ca > 0.5:
            reasons.append("Large discrepancy between claimed and approved amount")
    if not reasons:
        reasons.append("Statistical anomaly detected by ML model")
    return "; ".join(reasons)

# ─── Update Hospital Profiles ─────────────────────────────────────────────────
def _col(df: pd.DataFrame, name: str):
    """Case-insensitive column lookup. Returns the Series or None."""
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c]
    return None

def update_hospital_profiles(df: pd.DataFrame):
    conn = get_db()
    cur = conn.cursor()

    hosp_col = _col(df, "Hospital_ID")
    if hosp_col is None:
        conn.close()
        return

    work = df.copy()
    work["_hosp"]  = hosp_col.values
    risk_s = _col(df, "risk_score");  work["_risk"]   = risk_s.values   if risk_s   is not None else 50.0
    level_s = _col(df, "Risk_Level"); work["_level"]  = level_s.values  if level_s  is not None else "Medium"
    amt_s = _col(df, "Claim_Amount"); work["_amount"] = pd.to_numeric(amt_s, errors="coerce").fillna(0).values if amt_s is not None else 0.0
    loc_s = _col(df, "Location")

    for h_id, grp in work.groupby("_hosp"):
        total    = len(grp)
        flagged  = int(grp["_level"].isin(["High","Medium"]).sum())
        avg_risk = float(grp["_risk"].mean())
        t_amt    = float(grp["_amount"].sum())
        s_amt    = float(grp.loc[grp["_level"] == "High", "_amount"].sum())
        rl       = "High" if avg_risk >= 70 else ("Medium" if avg_risk >= 40 else "Low")
        if loc_s is not None:
            idx = grp.index[0]
            loc = str(loc_s.iloc[idx]) if idx < len(loc_s) else "Unknown"
        else:
            loc = "Unknown"

        cur.execute("""
            INSERT INTO hospitals
              (hospital_id, name, location, total_claims, flagged_claims,
               avg_risk_score, risk_level, total_amount, suspicious_amount, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(hospital_id) DO UPDATE SET
              total_claims      = total_claims + excluded.total_claims,
              flagged_claims    = flagged_claims + excluded.flagged_claims,
              avg_risk_score    = excluded.avg_risk_score,
              risk_level        = excluded.risk_level,
              total_amount      = total_amount + excluded.total_amount,
              suspicious_amount = suspicious_amount + excluded.suspicious_amount,
              updated_at        = excluded.updated_at
        """, (str(h_id), f"Hospital {h_id}", loc,
              total, flagged, avg_risk, rl, t_amt, s_amt,
              datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


# ─── Routes: Pages ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/investigate", response_class=HTMLResponse)
def investigate_page(request: Request):
    return templates.TemplateResponse("investigate.html", {"request": request})

@app.get("/hospitals", response_class=HTMLResponse)
def hospitals_page(request: Request):
    return templates.TemplateResponse("hospitals.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

# ─── Routes: Auth API ─────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=? AND password_hash=?",
                        (username, _hash_password(password))).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _make_token(username, user["role"])
    return {"token": token, "username": username, "role": user["role"]}

@app.get("/api/me")
async def me(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = _verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return payload

# ─── Routes: Upload & Predict ─────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_predict(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        df = run_prediction(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    batch_id = str(uuid.uuid4())[:8]
    conn = get_db()
    c = conn.cursor()

    col_map = {col.lower().replace(" ", "_"): col for col in df.columns}

    def gcol(name):
        return df.get(col_map.get(name), pd.Series([None]*len(df)))

    for _, row in df.iterrows():
        rs = float(row.get("risk_score", 0))
        rl = str(row.get("Risk_Level", "Low"))
        reasons = generate_fraud_reasons(row.to_dict(), df)
        c.execute("""INSERT INTO claims (upload_batch, claim_id, abha_id, patient_id, hospital_id, doctor_id,
                     diagnosis_code, treatment_code, admission_date, discharge_date,
                     claim_amount, approved_amount, location, risk_score, risk_level,
                     anomaly_score, fraud_reasons)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (batch_id,
                   str(row.get("Claim_ID", row.get("claim_id", ""))),
                   str(row.get("ABHA_ID", row.get("abha_id", row.get("ABHA_Id", "")))),
                   str(row.get("Patient_ID", row.get("patient_id", ""))),
                   str(row.get("Hospital_ID", row.get("hospital_id", ""))),
                   str(row.get("Doctor_ID", row.get("doctor_id", ""))),
                   str(row.get("Diagnosis_Code", row.get("diagnosis_code", ""))),
                   str(row.get("Treatment_Code", row.get("treatment_code", ""))),
                   str(row.get("Admission_Date", row.get("admission_date", ""))),
                   str(row.get("Discharge_Date", row.get("discharge_date", ""))),
                   float(row.get("Claim_Amount", row.get("claim_amount", 0)) or 0),
                   float(row.get("Approved_Amount", row.get("approved_amount", 0)) or 0),
                   str(row.get("Location", row.get("location", ""))),
                   rs, rl,
                   float(row.get("anomaly_score", 0) or 0),
                   reasons))
        if rs >= 70:
            c.execute("INSERT INTO alerts (claim_id, hospital_id, risk_score, message) VALUES (?,?,?,?)",
                      (str(row.get("Claim_ID", "")),
                       str(row.get("Hospital_ID", "")),
                       rs,
                       f"High-risk claim detected: score {rs:.1f}"))

    high = len(df[df["Risk_Level"] == "High"])
    medium = len(df[df["Risk_Level"] == "Medium"])
    low = len(df[df["Risk_Level"] == "Low"])
    c.execute("INSERT INTO upload_batches (batch_id, filename, total_claims, high_risk, medium_risk, low_risk) VALUES (?,?,?,?,?,?)",
              (batch_id, file.filename, len(df), high, medium, low))
    conn.commit()
    conn.close()

    update_hospital_profiles(df)

    result = {
        "batch_id": batch_id,
        "total": len(df),
        "high": high,
        "medium": medium,
        "low": low,
        "event": "upload_complete"
    }
    await manager.broadcast(result)
    return result

# ─── Routes: Dashboard Stats ──────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    conn = get_db()
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    flagged = c.execute("SELECT COUNT(*) FROM claims WHERE risk_level IN ('High','Medium')").fetchone()[0]
    suspicious_amt = c.execute("SELECT COALESCE(SUM(claim_amount),0) FROM claims WHERE risk_level='High'").fetchone()[0]
    unread_alerts = c.execute("SELECT COUNT(*) FROM alerts WHERE is_read=0").fetchone()[0]

    top_hospitals = c.execute("""
        SELECT hospital_id, COUNT(*) as cnt, AVG(risk_score) as avg_rs, SUM(claim_amount) as total_amt
        FROM claims WHERE risk_level='High'
        GROUP BY hospital_id ORDER BY cnt DESC LIMIT 5
    """).fetchall()

    trend = c.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as total,
               SUM(CASE WHEN risk_level='High' THEN 1 ELSE 0 END) as high
        FROM claims GROUP BY DATE(created_at) ORDER BY day
    """).fetchall()

    risk_dist = c.execute("""
        SELECT risk_level, COUNT(*) as cnt FROM claims GROUP BY risk_level
    """).fetchall()

    conn.close()
    return {
        "total_claims": total,
        "flagged_claims": flagged,
        "suspicious_amount": round(suspicious_amt, 2),
        "unread_alerts": unread_alerts,
        "top_hospitals": [dict(r) for r in top_hospitals],
        "trend": [dict(r) for r in trend],
        "risk_distribution": [dict(r) for r in risk_dist]
    }

# ─── Routes: Claims ───────────────────────────────────────────────────────────
@app.get("/api/claims")
def get_claims(hospital_id: str = "", risk_min: float = 0, date_from: str = "",
               date_to: str = "", page: int = 1, per_page: int = 50):
    conn = get_db()
    c = conn.cursor()
    where = ["1=1"]
    params = []
    if hospital_id:
        where.append("hospital_id=?")
        params.append(hospital_id)
    if risk_min:
        where.append("risk_score>=?")
        params.append(risk_min)
    if date_from:
        where.append("admission_date>=?")
        params.append(date_from)
    if date_to:
        where.append("admission_date<=?")
        params.append(date_to)
    where_str = " AND ".join(where)
    total = c.execute(f"SELECT COUNT(*) FROM claims WHERE {where_str}", params).fetchone()[0]
    offset = (page - 1) * per_page
    rows = c.execute(f"SELECT * FROM claims WHERE {where_str} ORDER BY risk_score DESC LIMIT ? OFFSET ?",
                     params + [per_page, offset]).fetchall()
    conn.close()
    return {"total": total, "page": page, "per_page": per_page,
            "claims": [dict(r) for r in rows]}

@app.get("/api/claims/{claim_id}/explain")
def explain_claim(claim_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Claim not found")
    d = dict(row)
    return {
        "claim": d,
        "explanation": d.get("fraud_reasons", "No explanation available"),
        "risk_score": d.get("risk_score", 0),
        "risk_level": d.get("risk_level", "Low")
    }

@app.patch("/api/claims/{claim_id}/status")
async def update_claim_status(claim_id: int, request: Request):
    body = await request.json()
    status = body.get("status", "Pending")
    conn = get_db()
    conn.execute("UPDATE claims SET status=? WHERE id=?", (status, claim_id))
    conn.commit()
    conn.close()
    return {"success": True}

# ─── Routes: Hospitals ────────────────────────────────────────────────────────
@app.get("/api/hospitals")
def get_hospitals():
    conn = get_db()
    rows = conn.execute("SELECT * FROM hospitals ORDER BY avg_risk_score DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/hospitals/{hospital_id}")
def get_hospital(hospital_id: str):
    conn = get_db()
    h = conn.execute("SELECT * FROM hospitals WHERE hospital_id=?", (hospital_id,)).fetchone()
    trend = conn.execute("""
        SELECT DATE(created_at) as day, AVG(risk_score) as avg_rs, COUNT(*) as cnt
        FROM claims WHERE hospital_id=? GROUP BY DATE(created_at) ORDER BY day
    """, (hospital_id,)).fetchall()
    claims = conn.execute("""
        SELECT * FROM claims WHERE hospital_id=? ORDER BY risk_score DESC LIMIT 10
    """, (hospital_id,)).fetchall()
    conn.close()
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return {
        "hospital": dict(h),
        "trend": [dict(r) for r in trend],
        "top_claims": [dict(r) for r in claims]
    }

# ─── Routes: Alerts ───────────────────────────────────────────────────────────
@app.get("/api/alerts")
def get_alerts():
    conn = get_db()
    rows = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int):
    conn = get_db()
    conn.execute("UPDATE alerts SET is_read=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return {"success": True}

# ─── Routes: Reports / CSV Export ────────────────────────────────────────────
@app.get("/api/reports/csv")
def export_csv(risk_level: str = ""):
    conn = get_db()
    if risk_level:
        rows = conn.execute("SELECT * FROM claims WHERE risk_level=? ORDER BY risk_score DESC", (risk_level,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM claims ORDER BY risk_score DESC").fetchall()
    conn.close()
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=fraud_report.csv"})

@app.get("/api/reports/summary")
def get_summary():
    conn = get_db()
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    high = c.execute("SELECT COUNT(*) FROM claims WHERE risk_level='High'").fetchone()[0]
    medium = c.execute("SELECT COUNT(*) FROM claims WHERE risk_level='Medium'").fetchone()[0]
    low = c.execute("SELECT COUNT(*) FROM claims WHERE risk_level='Low'").fetchone()[0]
    suspicious_amt = c.execute("SELECT COALESCE(SUM(claim_amount),0) FROM claims WHERE risk_level='High'").fetchone()[0]
    top_h = c.execute("""SELECT hospital_id, COUNT(*) as cnt, AVG(risk_score) as avg_rs
        FROM claims WHERE risk_level='High' GROUP BY hospital_id ORDER BY cnt DESC LIMIT 5""").fetchall()
    batches = c.execute("SELECT * FROM upload_batches ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return {
        "total": total, "high": high, "medium": medium, "low": low,
        "suspicious_amount": round(suspicious_amt, 2),
        "top_hospitals": [dict(r) for r in top_h],
        "batches": [dict(r) for r in batches]
    }

# ─── Routes: Settings ─────────────────────────────────────────────────────────
fraud_threshold = {"value": 70}

@app.get("/api/settings/threshold")
def get_threshold():
    return fraud_threshold

@app.post("/api/settings/threshold")
async def set_threshold(request: Request):
    body = await request.json()
    fraud_threshold["value"] = float(body.get("threshold", 70))
    return fraud_threshold

@app.delete("/api/settings/clear-data")
def clear_all_data():
    conn = get_db()
    c = conn.cursor()
    # Count before deleting for the response message
    claims_count   = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    hospital_count = c.execute("SELECT COUNT(*) FROM hospitals").fetchone()[0]
    alert_count    = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    batch_count    = c.execute("SELECT COUNT(*) FROM upload_batches").fetchone()[0]
    # Wipe all data tables (keep users)
    c.execute("DELETE FROM claims")
    c.execute("DELETE FROM hospitals")
    c.execute("DELETE FROM alerts")
    c.execute("DELETE FROM upload_batches")
    # Reset SQLite auto-increment counters
    c.execute("DELETE FROM sqlite_sequence WHERE name IN ('claims','hospitals','alerts','upload_batches')")
    conn.commit()
    conn.close()
    return {
        "success": True,
        "message": (
            f"Cleared {claims_count} claims, {hospital_count} hospital profiles, "
            f"{alert_count} alerts, and {batch_count} upload batches."
        )
    }

# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)