from pathlib import Path
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import os
import sqlite3
import hashlib
import hmac
import secrets
import re
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# -------------------- Paths and page setup --------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yield_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "combined_crop_yield.csv",
    ROOT / "data" / "processed" / "exploratory_model_dataset_for_handoff.csv",
]

st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- UI styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07130f; --panel:#10251b; --green:#75e0a1; --lime:#d0f078; --muted:#a7bdb0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background: radial-gradient(circle at 85% 0%,rgba(39,115,76,.20),transparent 27%),
              radial-gradient(circle at 0% 30%,rgba(65,128,82,.10),transparent 30%),var(--bg);
}
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10271c,#091810); border-right:1px solid #294b37; }
.hero {
  position:relative; overflow:hidden; padding:2rem 2.1rem; margin-bottom:1.2rem;
  border:1px solid rgba(117,224,161,.38); border-radius:24px;
  background:linear-gradient(120deg,rgba(24,69,47,.98),rgba(15,43,30,.96));
  box-shadow:0 18px 55px rgba(0,0,0,.22); animation:hero-in .65s ease-out both;
}
.hero:after { content:"🌱"; position:absolute; right:4%; top:-35px; font-size:8rem; opacity:.11; transform:rotate(-18deg); }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3rem); margin:0 0 .6rem; letter-spacing:-1px; }
.hero p { color:#d2e7d8; max-width:800px; margin:0; }
.kicker { color:var(--lime); font-size:.75rem; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; margin-bottom:.5rem; }
.metric-card {
  background:linear-gradient(145deg,#183c2a,#10251b); border:1px solid rgba(117,224,161,.27);
  border-radius:17px; padding:1.1rem 1.2rem; min-height:112px;
  box-shadow:0 10px 24px rgba(0,0,0,.14); transition:transform .2s,border-color .2s;
}
.metric-card:hover { transform:translateY(-4px) rotateX(1deg); border-color:rgba(208,240,120,.7); }
.metric-label { color:#b9d2c1; font-size:.82rem; margin-bottom:.45rem; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; color:#f4fff6; line-height:1.2; overflow-wrap:anywhere; }
.metric-note { color:#8db39a; font-size:.72rem; margin-top:.4rem; }
.panel { background:linear-gradient(145deg,rgba(21,48,34,.94),rgba(12,31,21,.96)); border:1px solid #294b37; border-radius:18px; padding:1.2rem 1.3rem; margin-bottom:.9rem; }
.stButton>button,.stDownloadButton>button { border-radius:11px; border:1px solid rgba(117,224,161,.35); background:linear-gradient(100deg,#72dfa0,#c7ec7c); color:#0a1a10; font-weight:700; min-height:2.6rem; transition:transform .2s,box-shadow .2s; }
.stButton>button:hover,.stDownloadButton>button:hover { color:#06120b; transform:translateY(-2px); box-shadow:0 8px 22px rgba(117,224,161,.18); }
div[data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid #294b37; }
button[data-baseweb="tab"] { background:rgba(18,42,29,.75); border-radius:10px 10px 0 0; padding:.65rem .85rem; }
button[data-baseweb="tab"][aria-selected="true"] { background:rgba(83,160,105,.18); border-bottom:2px solid var(--green); }
.small-muted { color:var(--muted); font-size:.88rem; }
@keyframes hero-in { from {opacity:0;transform:translateY(10px) rotateX(-2deg)} to {opacity:1;transform:translateY(0) rotateX(0)} }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important;transition:none!important; } }
.auth-wrap { max-width:1100px; margin:0 auto; }
.auth-hero { padding:1.8rem 2rem; margin:.5rem 0 1.2rem; border-radius:24px; border:1px solid rgba(117,224,161,.38); background:linear-gradient(120deg,#18452f,#10251b); box-shadow:0 18px 55px rgba(0,0,0,.2); }
.auth-hero h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3rem); margin:.2rem 0 .6rem; }
.auth-hero p { color:#d2e7d8; margin:0; max-width:850px; }
.auth-info { min-height:360px; padding:1.6rem; }
.auth-info h2 { font-family:'Space Grotesk',sans-serif; line-height:1.25; }
.auth-feature { padding:.75rem 0; border-bottom:1px solid #294b37; line-height:1.6; }
.auth-feature span { color:#a7bdb0; font-size:.9rem; padding-left:1.8rem; }
.auth-card { padding:1.5rem; }
</style>
""", unsafe_allow_html=True)


# -------------------- Local SQLite Authentication --------------------
# Local account database. Passwords are stored as salted PBKDF2 hashes, never as plain text.
AUTH_DB_PATH = ROOT / "data" / "agripredict_users.db"
AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_auth_connection():
    conn = sqlite3.connect(AUTH_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    with get_auth_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Student / Researcher',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 310_000)
    return digest.hex(), salt


def create_local_user(email, password, display_name):
    email = email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")
    if len(password) > 256:
        raise ValueError("Password is too long.")
    password_hash, salt = hash_password(password)
    name = display_name.strip() or email.split("@")[0]
    try:
        with get_auth_connection() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, password_salt, display_name) VALUES (?, ?, ?, ?)",
                (email, password_hash, salt, name[:80]),
            )
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists. Please sign in.") from None
    return email, name[:80], "Student / Researcher"


def verify_local_user(email, password):
    email = email.strip().lower()
    with get_auth_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        # Perform a dummy hash to reduce obvious timing differences for unknown emails.
        hash_password(password, "00112233445566778899aabbccddeeff")
        raise ValueError("Incorrect email or password.")
    calculated_hash, _ = hash_password(password, user["password_salt"])
    if not hmac.compare_digest(calculated_hash, user["password_hash"]):
        raise ValueError("Incorrect email or password.")
    return user["email"], user["display_name"], user["role"]


def update_local_profile(email, display_name, role):
    with get_auth_connection() as conn:
        conn.execute(
            "UPDATE users SET display_name = ?, role = ? WHERE email = ?",
            (display_name[:80], role, email),
        )


def get_local_user(email):
    with get_auth_connection() as conn:
        return conn.execute("SELECT email, display_name, role, created_at FROM users WHERE email = ?", (email,)).fetchone()


def local_sign_out():
    for key in ("auth_email", "profile_name", "profile_role"):
        st.session_state.pop(key, None)
    st.session_state["latest_prediction"] = None
    st.session_state["generated_pdf"] = None


def render_auth_page():
    """Require local SQLite account sign-in or registration before displaying the dashboard."""
    st.markdown("""
    <div class="auth-wrap">
      <div class="auth-hero">
        <div class="kicker">Smart farming · Data · Machine learning</div>
        <h1>🌾 AgriPredict AI</h1>
        <p>Your agricultural intelligence workspace. Sign in to explore crop-yield trends, test scenarios, and generate reports.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("""
        <div class="panel auth-info">
          <h2>Make data-informed exploration simpler.</h2>
          <p>One workspace for historical crop data, experimental yield estimates, transparent model evaluation, and PDF reporting.</p>
          <div class="auth-feature">🌱 <b>Yield prediction</b><br><span>Explore six-feature model estimates.</span></div>
          <div class="auth-feature">📊 <b>Historical insights</b><br><span>Compare crops and inspect dataset records.</span></div>
          <div class="auth-feature">📄 <b>Exportable reports</b><br><span>Generate a summary PDF for your project.</span></div>
          <p class="small-muted">Predictions are experimental and are not validated agricultural advice.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel auth-card">', unsafe_allow_html=True)
        st.markdown("### Welcome")
        st.caption("Create a local account or sign in with your existing account.")
        mode = st.radio("Authentication", ["Sign in", "Create account"], horizontal=True, label_visibility="collapsed")
        with st.form("local_auth_form", clear_on_submit=False):
            email = st.text_input("Email address", placeholder="you@example.com", autocomplete="email")
            password = st.text_input("Password", type="password", placeholder="At least 8 characters", autocomplete="current-password" if mode == "Sign in" else "new-password")
            display_name = ""
            if mode == "Create account":
                display_name = st.text_input("Display name", placeholder="Your name", max_chars=80)
            submitted = st.form_submit_button("Sign in" if mode == "Sign in" else "Create account", use_container_width=True, type="primary")
        if submitted:
            if not email.strip() or not password:
                st.error("Enter both your email address and password.")
            else:
                try:
                    if mode == "Create account":
                        user_email, name, role = create_local_user(email, password, display_name)
                        st.success("Account created successfully.")
                    else:
                        user_email, name, role = verify_local_user(email, password)
                    st.session_state.auth_email = user_email
                    st.session_state.profile_name = name
                    st.session_state.profile_role = role
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except sqlite3.Error:
                    st.error("The local account database could not be accessed. Check folder permissions and try again.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("This is local authentication for a project demo. Accounts are stored on this computer and are not shared across devices.")
    st.stop()


init_auth_db()
# Do not render dashboard content until a user has successfully signed in.
if not st.session_state.get("auth_email"):
    render_auth_page()

# Load saved profile values from the local account database on each app rerun.
current_user = get_local_user(st.session_state.auth_email)
if current_user:
    st.session_state.profile_name = current_user["display_name"]
    st.session_state.profile_role = current_user["role"]
else:
    local_sign_out()
    st.rerun()


# -------------------- Helpers --------------------
def load_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = [str(c).strip() for c in df.columns]
                if not df.empty and {"crop", "year"}.issubset(df.columns):
                    return df, path
            except Exception:
                pass
    return pd.DataFrame(), None


def find_column(df, choices):
    return next((name for name in choices if name in df.columns), None)


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def read_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        frame = pd.read_csv(METRICS_PATH)
        return frame.iloc[0] if not frame.empty else None
    except Exception:
        return None


def metric_value(row, *names):
    if row is None:
        return None
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return None


def create_pdf(report):
    if not PDF_AVAILABLE:
        return None
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AgriPredict AI — Agricultural Analytics Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph("Project overview", styles["Heading2"]),
        Paragraph("AgriPredict AI is an experimental project for exploring historical crop-yield patterns and demonstrating a machine-learning workflow.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Dataset summary", styles["Heading2"]),
    ]
    rows = [
        ["Item", "Value"],
        ["Records", str(report["records"])],
        ["Crops", report["crops"]],
        ["Year range", report["years"]],
        ["Dataset file", report["dataset_file"]],
    ]
    table = Table(rows, colWidths=[150, 330])
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18452f")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#9bb6a3")),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f4f8f4")),
        ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#183326")),
        ("PADDING",(0,0),(-1,-1),7),
    ]))
    story.extend([table, Spacer(1, 14), Paragraph("Latest prediction", styles["Heading2"])])
    pred = report.get("prediction")
    if pred:
        p_rows = [
            ["Item", "Value"],
            ["Crop", str(pred["crop"])],
            ["Year", str(pred["year"])],
            ["Estimated yield", f'{pred["yield"]:.3f} tonnes/hectare'],
            ["Annual rainfall", f'{pred["rainfall"]:.2f} mm'],
            ["Maximum temperature", f'{pred["tmax"]:.2f} °C'],
            ["Mean temperature", f'{pred["tmean"]:.2f} °C'],
            ["Minimum temperature", f'{pred["tmin"]:.2f} °C'],
        ]
        ptable = Table(p_rows, colWidths=[150, 330])
        ptable.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18452f")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#9bb6a3")),
            ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f4f8f4")),
            ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#183326")),
            ("PADDING",(0,0),(-1,-1),7),
        ]))
        story.append(ptable)
    else:
        story.append(Paragraph("No prediction generated during this session.", styles["BodyText"]))
    story.extend([Spacer(1, 14), Paragraph("Model evaluation", styles["Heading2"])])
    m = report.get("metrics")
    if m:
        m_rows = [["Metric", "Value"], ["Model", m["model"]], ["Test MAE", m["mae"]],
                  ["Test RMSE", m["rmse"]], ["Test R²", m["r2"]],
                  ["Test period", m["period"]], ["Test records", m["rows"]]]
        mtable = Table(m_rows, colWidths=[150, 330])
        mtable.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18452f")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#9bb6a3")),
            ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f4f8f4")),
            ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#183326")),
            ("PADDING",(0,0),(-1,-1),7),
        ]))
        story.append(mtable)
    else:
        story.append(Paragraph("Evaluation metrics are unavailable.", styles["BodyText"]))
    story.extend([
        Spacer(1, 16), Paragraph("Limitations and responsible use", styles["Heading2"]),
        Paragraph("This model is experimental. Historical associations do not guarantee future yield. Data may differ in geographic coverage and may not represent local farm conditions. Do not use this report as the sole basis for planting, pesticide, or financial decisions.", styles["BodyText"]),
    ])
    doc.build(story)
    output.seek(0)
    return output.getvalue()


# -------------------- Load data/model --------------------
data, data_path = load_dataset()
if not data.empty:
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    if "yield_tonnes_per_hectare" in data.columns:
        data["yield_tonnes_per_hectare"] = pd.to_numeric(data["yield_tonnes_per_hectare"], errors="coerce")
    data = data.dropna(subset=["year", "crop"])
    data["year"] = data["year"].astype(int)
    data["crop"] = data["crop"].astype(str).str.strip()

crops = sorted(data["crop"].dropna().unique().tolist()) if not data.empty else []
first_year = int(data["year"].min()) if not data.empty else 1991
last_year = int(data["year"].max()) if not data.empty else 2024
metrics = read_metrics()

model, model_error = None, None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        model_error = str(exc)

for key, default in [
    ("profile_name", "AgriPredict User"),
    ("profile_role", "Student / Researcher"),
    ("latest_prediction", None),
    ("generated_pdf", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------- Sidebar and header --------------------
with st.sidebar:
    st.markdown("## 🌾 AgriPredict AI")
    st.caption("Agricultural intelligence workspace")
    st.divider()
    st.markdown("### Dataset at a glance")
    if not data.empty:
        st.metric("Records", f"{len(data):,}")
        st.metric("Crop varieties", len(crops))
        st.caption(f"Years covered: {first_year}–{last_year}")
    else:
        st.warning("Processed dataset not found.")
    st.divider()
    st.markdown("### Workspace profile")
    st.write(st.session_state.profile_name)
    st.caption(st.session_state.profile_role)
    st.caption(st.session_state.get("auth_email", "Signed-in user"))
    if st.button("🚪 Sign out", use_container_width=True):
        local_sign_out()
        st.rerun()
    st.divider()
    st.caption("⚠️ Experimental model — not validated agricultural advice.")

st.markdown("""
<div class="hero">
  <div class="kicker">Smart farming · Data · Machine learning</div>
  <h1>🌾 AgriPredict AI</h1>
  <p>Explore crop-yield trends, test weather scenarios, review model evaluation, and export a report in one workspace.</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_predict, tab_explore, tab_performance, tab_report, tab_profile, tab_about = st.tabs([
    "🏠 Overview", "🌱 Yield Prediction", "📊 Explore Data",
    "🧠 Model Performance", "📄 Generate Report", "👤 Profile", "ℹ️ About"
])


# -------------------- Overview tab --------------------
with tab_overview:
    st.markdown('<div class="kicker">Your agricultural workspace</div>', unsafe_allow_html=True)
    st.title("Welcome to your dashboard")
    st.write("Explore the dataset first, then experiment with a yield estimate.")
    if data.empty:
        st.error("No valid processed dataset found. Check data/processed/ and ensure the CSV has 'year' and 'crop' columns.")
    else:
        avg_yield = data["yield_tonnes_per_hectare"].mean() if "yield_tonnes_per_hectare" in data.columns else np.nan
        cols = st.columns(4)
        with cols[0]: metric_card("Dataset records", f"{len(data):,}", "Rows available")
        with cols[1]: metric_card("Crop varieties", str(len(crops)), "Distinct crop labels")
        with cols[2]: metric_card("First year", str(first_year), "In loaded dataset")
        with cols[3]: metric_card("Latest year", str(last_year), "In loaded dataset")
        st.markdown("### Crop yield snapshot")
        if "yield_tonnes_per_hectare" in data.columns:
            means = data.groupby("crop", as_index=False)["yield_tonnes_per_hectare"].mean().dropna()
            if not means.empty:
                left, right = st.columns(2)
                with left:
                    fig = px.bar(means, x="crop", y="yield_tonnes_per_hectare", color="crop",
                                 title="Average recorded yield by crop",
                                 labels={"crop":"Crop", "yield_tonnes_per_hectare":"Average yield (tonnes/hectare)"},
                                 template="plotly_dark")
                    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    counts = data["crop"].value_counts().rename_axis("crop").reset_index(name="records")
                    fig2 = px.pie(counts, names="crop", values="records", hole=.62,
                                  title="Records by crop", template="plotly_dark")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="panel"><h3>🌿 Getting started</h3><p class="small-muted">Use Explore Data for historical records, Yield Prediction to test a scenario, Model Performance to review evaluation, and Generate Report to export a PDF.</p></div>', unsafe_allow_html=True)
        st.info("Predictions are experimental and should not be treated as validated agricultural advice.")


# -------------------- Prediction tab --------------------
with tab_predict:
    st.markdown('<div class="kicker">Scenario lab</div>', unsafe_allow_html=True)
    st.title("🌱 Crop yield prediction")
    st.write("Choose a crop, year, and annual weather values to generate an experimental estimate.")
    if model is None:
        st.error("Trained model could not be loaded. Confirm models/yield_model.joblib exists.")
        if model_error: st.code(model_error)
    elif data.empty or not crops:
        st.warning("A dataset with crop labels is required.")
    else:
        with st.form("prediction_form"):
            a, b = st.columns(2)
            with a: chosen_crop = st.selectbox("Select crop", crops)
            with b: pred_year = st.number_input("Prediction year", min_value=1, max_value=2100, value=min(last_year + 1, 2100), step=1)
            st.markdown("#### 🌦️ Annual weather inputs")
            crop_data = data[data["crop"].str.casefold() == chosen_crop.casefold()]
            rain_col = find_column(data, ["rainfall_ann", "annual_rainfall_mm", "rainfall_mm"])
            max_col = find_column(data, ["tmax_ann", "annual_tmax", "tmax"])
            mean_col = find_column(data, ["tmean_ann", "annual_tmean", "tmean", "avg_temperature_c"])
            min_col = find_column(data, ["tmin_ann", "annual_tmin", "tmin"])

            def median_default(col, default):
                if col:
                    values = pd.to_numeric(crop_data[col], errors="coerce").dropna()
                    if not values.empty: return float(values.median())
                return float(default)

            rain_default = median_default(rain_col, 850)
            max_default = median_default(max_col, 31)
            mean_default = median_default(mean_col, 25)
            min_default = median_default(min_col, 20)
            w1, w2 = st.columns(2)
            with w1:
                rainfall = st.number_input("Annual rainfall (mm)", min_value=0.0, max_value=20000.0, value=max(0.0, rain_default), step=10.0)
                tmax = st.number_input("Annual maximum temperature (°C)", min_value=-30.0, max_value=70.0, value=float(np.clip(max_default, -30, 70)), step=.5)
            with w2:
                tmean = st.number_input("Annual mean temperature (°C)", min_value=-30.0, max_value=70.0, value=float(np.clip(mean_default, -30, 70)), step=.5)
                tmin = st.number_input("Annual minimum temperature (°C)", min_value=-50.0, max_value=60.0, value=float(np.clip(min_default, -50, 60)), step=.5)
            submitted = st.form_submit_button("🌾 Generate yield estimate", use_container_width=True)
        if submitted:
            try:
                input_df = pd.DataFrame([{
                    "year": int(pred_year), "crop": chosen_crop, "rainfall_ann": float(rainfall),
                    "tmax_ann": float(tmax), "tmean_ann": float(tmean), "tmin_ann": float(tmin)
                }], columns=["year","crop","rainfall_ann","tmax_ann","tmean_ann","tmin_ann"])
                predicted = float(model.predict(input_df)[0])
                if not np.isfinite(predicted): raise ValueError("Model returned a non-finite value.")
                st.session_state.latest_prediction = {
                    "crop": chosen_crop, "year": int(pred_year), "yield": predicted,
                    "rainfall": float(rainfall), "tmax": float(tmax),
                    "tmean": float(tmean), "tmin": float(tmin),
                    "generated_at": datetime.now().isoformat(timespec="minutes")
                }
            except Exception as exc:
                st.error("Prediction failed. Check that the saved model was trained with these six features.")
                st.code(str(exc))
        latest = st.session_state.latest_prediction
        if latest:
            st.markdown("### Latest estimate")
            a, b, c = st.columns(3)
            with a: metric_card("Selected crop", latest["crop"], f'Year: {latest["year"]}')
            with b: metric_card("Estimated yield", f'{latest["yield"]:.3f}', "tonnes/hectare")
            with c: metric_card("Annual rainfall", f'{latest["rainfall"]:.1f} mm', "Scenario input")
            if latest["year"] > last_year:
                st.warning("This year is beyond the dataset range. This is an extrapolation and may be unreliable.")
            st.info("This estimate is not a validated farm-level forecast. Do not use it alone for planting, pesticide, or financial decisions.")


# -------------------- Explore Data tab --------------------
with tab_explore:
    st.markdown('<div class="kicker">Dataset explorer</div>', unsafe_allow_html=True)
    st.title("📊 Explore historical crop yields")
    st.write("Filter records, compare crop trends, and download the filtered dataset.")
    if data.empty:
        st.warning("Dataset unavailable.")
    else:
        f1, f2 = st.columns(2)
        with f1: chosen_crops = st.multiselect("Choose crops", crops, default=crops)
        with f2: years = st.slider("Choose year range", min_value=first_year, max_value=last_year, value=(first_year,last_year))
        filtered = data[data["crop"].isin(chosen_crops) & data["year"].between(years[0], years[1])].copy()
        avg = filtered["yield_tonnes_per_hectare"].mean() if "yield_tonnes_per_hectare" in filtered.columns and not filtered.empty else np.nan
        a,b,c = st.columns(3)
        with a: metric_card("Filtered records", f"{len(filtered):,}", "After filters")
        with b: metric_card("Crops selected", str(len(chosen_crops)), "Crop categories")
        with c: metric_card("Average yield", f"{avg:.3f}" if pd.notna(avg) else "N/A", "tonnes/hectare")
        if not filtered.empty and "yield_tonnes_per_hectare" in filtered.columns:
            fig = px.line(filtered.sort_values("year"), x="year", y="yield_tonnes_per_hectare", color="crop",
                          markers=True, title="Historical yield trends by crop",
                          labels={"year":"Year","yield_tonnes_per_hectare":"Yield (tonnes/hectare)","crop":"Crop"},
                          template="plotly_dark")
            fig.update_layout(hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No yield values available for this filter selection.")
        st.markdown("### Dataset records")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download filtered dataset (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                           file_name="agripredict_filtered_data.csv", mime="text/csv")


# -------------------- Model Performance tab --------------------
with tab_performance:
    st.markdown('<div class="kicker">Evaluation center</div>', unsafe_allow_html=True)
    st.title("🧠 Model performance")
    st.write("Review the evaluation metrics saved by the training pipeline.")
    if metrics is None:
        st.warning("Metrics unavailable. Check reports/model_metrics.csv.")
    else:
        mae = metric_value(metrics, "Test_MAE", "test_MAE", "test_mae", "MAE", "mae")
        rmse = metric_value(metrics, "Test_RMSE", "test_RMSE", "test_rmse", "RMSE", "rmse")
        r2 = metric_value(metrics, "Test_R2", "test_R2", "test_r2", "R2", "r2", "r_squared")
        model_name = metrics.get("Model", metrics.get("model", "Not specified"))
        start = metrics.get("Test_Start_Year", metrics.get("test_start_year", None))
        end = metrics.get("Test_End_Year", metrics.get("test_end_year", None))
        rows = metrics.get("Test_Rows", metrics.get("test_rows", None))
        a,b,c = st.columns(3)
        with a: metric_card("Test MAE", f"{mae:.3f}" if mae is not None else "N/A", "Lower is generally better")
        with b: metric_card("Test RMSE", f"{rmse:.3f}" if rmse is not None else "N/A", "Penalizes larger errors")
        with c: metric_card("Test R²", f"{r2:.3f}" if r2 is not None else "N/A", "Closer to 1 can indicate better fit")
        details = [f"Selected model: {model_name}"]
        if pd.notna(start) and pd.notna(end): details.append(f"Test period: {int(start)}–{int(end)}")
        if pd.notna(rows): details.append(f"Test records: {int(rows)}")
        st.caption(" · ".join(details))
        if mae is None or rmse is None or r2 is None:
            st.warning("Some metric headers were not recognized. Check reports/model_metrics.csv.")
        pred_path = ROOT / "reports" / "test_predictions.csv"
        if pred_path.exists():
            try:
                p_df = pd.read_csv(pred_path)
                actual_col = find_column(p_df, ["actual","y_true","actual_yield","yield_tonnes_per_hectare"])
                predicted_col = find_column(p_df, ["predicted","y_pred","predicted_yield","prediction"])
                if actual_col and predicted_col:
                    st.markdown("### Actual vs predicted")
                    fig = px.scatter(p_df, x=actual_col, y=predicted_col,
                                     color="crop" if "crop" in p_df.columns else None,
                                     labels={actual_col:"Actual yield (tonnes/hectare)",predicted_col:"Predicted yield (tonnes/hectare)"},
                                     title="Test-set predictions compared with actual yields", template="plotly_dark")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.caption(f"Could not load prediction plot: {exc}")
        st.warning("These scores describe the saved test split only. Geographic coverage and data quality limit reliability; the model is not validated for pre-season agricultural advice.")


# -------------------- PDF report tab --------------------
with tab_report:
    st.markdown('<div class="kicker">Export center</div>', unsafe_allow_html=True)
    st.title("📄 Generate report")
    st.write("Build a PDF containing dataset details, the latest prediction, model metrics, and limitations.")
    if st.session_state.latest_prediction:
        p = st.session_state.latest_prediction
        st.success(f'Latest estimate: {p["crop"]}, {p["year"]} — {p["yield"]:.3f} tonnes/hectare.')
    else:
        st.info("No prediction generated in this session. You can still create a project summary report.")
    if not PDF_AVAILABLE:
        st.error("PDF support requires ReportLab. Run: python -m pip install reportlab")
    else:
        if st.button("🧾 Build PDF report", use_container_width=True):
            metric_report = None
            if metrics is not None:
                mae = metric_value(metrics, "Test_MAE", "test_MAE", "test_mae", "MAE", "mae")
                rmse = metric_value(metrics, "Test_RMSE", "test_RMSE", "test_rmse", "RMSE", "rmse")
                r2 = metric_value(metrics, "Test_R2", "test_R2", "test_r2", "R2", "r2")
                model_name = metrics.get("Model", metrics.get("model", "N/A"))
                start = metrics.get("Test_Start_Year", metrics.get("test_start_year", None))
                end = metrics.get("Test_End_Year", metrics.get("test_end_year", None))
                rows = metrics.get("Test_Rows", metrics.get("test_rows", None))
                metric_report = {
                    "model": str(model_name),
                    "mae": f"{mae:.3f}" if mae is not None else "N/A",
                    "rmse": f"{rmse:.3f}" if rmse is not None else "N/A",
                    "r2": f"{r2:.3f}" if r2 is not None else "N/A",
                    "period": f"{int(start)}–{int(end)}" if pd.notna(start) and pd.notna(end) else "N/A",
                    "rows": str(int(rows)) if pd.notna(rows) else "N/A",
                }
            report = {
                "records": len(data) if not data.empty else 0,
                "crops": ", ".join(crops) if crops else "N/A",
                "years": f"{first_year}–{last_year}" if not data.empty else "N/A",
                "dataset_file": data_path.name if data_path else "Not found",
                "prediction": st.session_state.latest_prediction,
                "metrics": metric_report,
            }
            try:
                st.session_state.generated_pdf = create_pdf(report)
                if st.session_state.generated_pdf:
                    st.success("PDF report generated. Use the download button below.")
                else:
                    st.error("PDF generation failed.")
            except Exception as exc:
                st.error("PDF generation failed.")
                st.code(str(exc))
        if st.session_state.generated_pdf:
            st.download_button("⬇️ Download AgriPredict AI report (PDF)",
                               data=st.session_state.generated_pdf,
                               file_name="AgriPredict_AI_Report.pdf", mime="application/pdf",
                               use_container_width=True)


# -------------------- Profile tab --------------------
with tab_profile:
    st.markdown('<div class="kicker">Workspace settings</div>', unsafe_allow_html=True)
    st.title("👤 Profile")
    st.write("Personalize the display name and role shown in this dashboard.")
    roles = ["Student / Researcher", "Farmer", "Agriculture analyst", "Faculty / Mentor", "Other"]
    with st.form("profile_form"):
        name = st.text_input("Display name", value=st.session_state.profile_name, max_chars=60)
        current_role = st.session_state.profile_role if st.session_state.profile_role in roles else roles[0]
        role = st.selectbox("Role", roles, index=roles.index(current_role))
        save = st.form_submit_button("Save profile")
    if save:
        st.session_state.profile_name = name.strip() or "AgriPredict User"
        st.session_state.profile_role = role
        update_local_profile(st.session_state.auth_email, st.session_state.profile_name, st.session_state.profile_role)
        st.success("Profile updated and saved locally.")
    st.markdown(f'<div class="panel"><h3>🔐 Account information</h3><p><b>Signed-in email:</b> {st.session_state.get("auth_email", "Unavailable")}</p><p class="small-muted">Authentication uses a local SQLite database. Passwords are stored as salted hashes, not plain text. Profile changes are saved on this computer.</p></div>', unsafe_allow_html=True)


# -------------------- About tab --------------------
with tab_about:
    st.markdown('<div class="kicker">Project documentation</div>', unsafe_allow_html=True)
    st.title("ℹ️ About AgriPredict AI")
    st.markdown('<div class="panel"><h3>🌾 Project overview</h3><p>AgriPredict AI is an experimental agricultural analytics project for exploring historical crop-yield records and demonstrating a machine-learning workflow.</p></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### 🎯 Objectives")
        st.markdown("- Explore historical crop-yield trends.\n- Compare crop categories.\n- Experiment with weather-related features.\n- Present evaluation metrics transparently.\n- Export a project report.")
        st.markdown("### 🧰 Technology stack")
        st.markdown("- Python and Pandas\n- Scikit-learn and Joblib\n- Streamlit\n- Plotly\n- ReportLab for PDF export")
    with right:
        st.markdown("### 📁 Dataset information")
        st.write(f"**Records:** {len(data):,}" if not data.empty else "**Records:** Not available")
        st.write(f"**Crops:** {', '.join(crops)}" if crops else "**Crops:** Not available")
        st.write(f"**Year range:** {first_year}–{last_year}" if not data.empty else "**Year range:** Not available")
        st.write("**Prediction target:** Yield in tonnes per hectare")
        st.markdown("### ⚠️ Limitations")
        st.markdown("- Sources may have different geographic coverage.\n- Historical relationships do not guarantee future outcomes.\n- Future-year estimates are extrapolations.\n- Weather values may not represent a specific farm or season.\n- Independent validation is needed before real-world use.\n- Do not use this tool as the sole basis for farming or pesticide decisions.")
    st.caption("AgriPredict AI · Educational and experimental decision-support prototype")
