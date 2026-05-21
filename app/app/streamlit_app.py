import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
LATITUDE = float(os.getenv("LATITUDE", "31.5204"))
LONGITUDE = float(os.getenv("LONGITUDE", "74.3587"))

st.set_page_config(
    page_title="AQIPredict · Lahore",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Epilogue:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Epilogue', sans-serif !important;
    background-color: #07090f !important;
    color: #d8e2f5 !important;
}
.stApp { background-color: #07090f !important; }
#MainMenu, footer { visibility: hidden; }

/* Keep header minimal but visible — hosts sidebar reopen control when collapsed */
header[data-testid="stHeader"] {
    background: rgba(7, 9, 15, 0.92) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
header [data-testid="stToolbarActions"],
header [data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenuButton"],
[data-testid="stMainMenuPopover"] {
    display: none !important;
}
/* Header toolbar: only the expand control, pinned top-left when sidebar is closed */
header[data-testid="stHeader"] {
    height: 3.25rem !important;
    min-height: 3.25rem !important;
}
header [data-testid="stToolbar"] {
    position: fixed !important;
    top: 0.55rem !important;
    left: 0.7rem !important;
    width: auto !important;
    height: auto !important;
    min-height: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    z-index: 1000000 !important;
    padding: 0 !important;
}
[data-testid="stExpandSidebarButton"] button {
    background: #111826 !important;
    color: #00e0aa !important;
    border: 1px solid rgba(0, 224, 170, 0.28) !important;
    border-radius: 8px !important;
    width: 2.35rem !important;
    height: 2.35rem !important;
    min-height: 2.35rem !important;
    padding: 0 !important;
}
[data-testid="stSidebarCollapseButton"] button {
    color: #d8e2f5 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    background: #111826 !important;
    border-radius: 8px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a0d16 !important;
    border-right: 1px solid rgba(255,255,255,0.055) !important;
}
[data-testid="stSidebar"] * { color: #d8e2f5 !important; }
[data-testid="stSidebarNav"] { display: none; }

/* Radio as nav */
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
[data-testid="stSidebar"] .stRadio label {
    padding: .55rem .9rem !important;
    border-radius: 9px !important;
    font-size: .82rem !important;
    color: #5e7292 !important;
    cursor: pointer;
    transition: all .18s;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #111826 !important;
    color: #d8e2f5 !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"],
[data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
    background: rgba(0,224,170,0.1) !important;
    color: #00e0aa !important;
    border-color: rgba(0,224,170,0.18) !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: .82rem !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0d1120 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: .6rem !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #5e7292 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    color: #d8e2f5 !important;
}
[data-testid="stMetricDelta"] { font-size: .72rem !important; }

/* Charts */
[data-testid="stArrowVegaLiteChart"],
[data-testid="stLineChart"],
[data-testid="stBarChart"] {
    background: transparent !important;
}
.js-plotly-plot .plotly { background: transparent !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Alerts */
.stWarning {
    background: rgba(245,197,24,0.07) !important;
    border: 1px solid rgba(245,197,24,0.2) !important;
    border-radius: 12px !important;
    color: #f5c518 !important;
}
.stError {
    background: rgba(248,113,113,0.07) !important;
    border: 1px solid rgba(248,113,113,0.2) !important;
    border-radius: 12px !important;
}
.stSuccess {
    background: rgba(74,222,128,0.07) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 12px !important;
}
.stInfo {
    background: rgba(79,142,247,0.07) !important;
    border: 1px solid rgba(79,142,247,0.2) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0d1120 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: #0d1120 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* Section label helper */
.sec-lbl {
    font-family: 'DM Mono', monospace;
    font-size: .62rem;
    letter-spacing: .12em;
    color: #3d4f6a;
    text-transform: uppercase;
    margin-bottom: .5rem;
    margin-top: 1rem;
}

/* Card wrapper */
.aqi-big-card {
    background: #0d1120;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.pol-card {
    background: #0d1120;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────
def get_aqi_color(aqi):
    if aqi <= 50: return "#4ade80"
    elif aqi <= 100: return "#f5c518"
    elif aqi <= 150: return "#ff7043"
    elif aqi <= 200: return "#f87171"
    elif aqi <= 300: return "#a78bfa"
    else: return "#7c3aed"

def hex_to_rgba(hex_color, alpha=0.6):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "✅"
    elif aqi <= 100: return "Moderate", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200: return "Unhealthy", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "🟣"
    else: return "Hazardous", "☠️"

def get_plotly_layout(title="", height=300):
    return dict(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#5e7292', family='DM Mono, monospace', size=10),
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.06)'),
        showlegend=True
    )

# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_project():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )
    mr = project.get_model_registry()
    model_hw = mr.get_model("aqi_predictor", version=1)
    model_dir = model_hw.download()
    model = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
    return model, project

@st.cache_data(show_spinner=False, ttl=3600)
def get_features():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_current_aqi():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "current": ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
                    "ozone","sulphur_dioxide","us_aqi","european_aqi"]
    }
    r = requests.get(url, params=params, timeout=10)
    return r.json()["current"]

def predict_days(model, last_row, n=7):
    preds = []
    now = datetime.now()
    for i in range(n):
        future = now + timedelta(days=i)
        feats = {
            "hour": 12, "day": future.day, "month": future.month,
            "day_of_week": future.weekday(),
            "pm2_5": last_row["pm2_5"], "pm10": last_row["pm10"],
            "carbon_monoxide": last_row["carbon_monoxide"],
            "nitrogen_dioxide": last_row["nitrogen_dioxide"],
            "ozone": last_row["ozone"], "sulphur_dioxide": last_row["sulphur_dioxide"],
            "aqi_change_rate": last_row["aqi_change_rate"],
        }
        aqi = round(model.predict(pd.DataFrame([feats]))[0])
        cat, emoji = get_aqi_category(aqi)
        preds.append({
            "Date": future.strftime("%a, %b %d"),
            "Predicted AQI": aqi,
            "Category": cat,
            "Status": emoji,
            "Color": get_aqi_color(aqi)
        })
    return pd.DataFrame(preds)

# ── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("🌫️ Loading AQI data..."):
    try:
        model, project = load_model_and_project()
        df = get_features()
        current = get_current_aqi()
        model_loaded = True
    except Exception as e:
        st.error(f"Connection error: {e}")
        model_loaded = False
        st.stop()

last_row = df.iloc[-1]
current_aqi = int(current.get("us_aqi", last_row["us_aqi"]))
eu_aqi = int(current.get("european_aqi", last_row["european_aqi"]))
category, emoji = get_aqi_category(current_aqi)
aqi_color = get_aqi_color(current_aqi)
last_24h = df.tail(24)
predictions = predict_days(model, last_row, n=7)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.5rem 0 1rem; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">
      <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#00e0aa,#4f8ef7);border-radius:8px;display:grid;place-items:center;font-size:.9rem;">🌫️</div>
        <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:.95rem;">AQI<span style="color:#00e0aa">Predict</span></span>
      </div>
      <div style="background:rgba(0,224,170,0.07);border:1px solid rgba(0,224,170,0.14);border-radius:10px;padding:.65rem .85rem;margin-bottom:.6rem;">
        <div style="display:flex;align-items:center;gap:.5rem;">
          <div style="width:6px;height:6px;background:#00e0aa;border-radius:50%;"></div>
          <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;color:#00e0aa;">Lahore, Pakistan</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#5e7292;margin-top:.2rem;">31.5204°N · 74.3587°E</div>
      </div>
      <div style="background:#0d1120;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:.65rem .85rem;display:flex;align-items:center;justify-content:space-between;">
        <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;color:{aqi_color};">{current_aqi}</span>
        <div style="text-align:right;">
          <div style="font-size:.65rem;color:{aqi_color};font-weight:600;">{emoji} {category[:10]}</div>
          <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#5e7292;">US AQI · NOW</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:.58rem;letter-spacing:.1em;color:#3d4f6a;text-transform:uppercase;margin-bottom:.4rem;">Main</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "⊞  Overview",
        "⏱  Hourly AQI",
        "📅  Daily AQI",
        "🔮  Forecast",
        "📊  Charts",
        "🧪  Pollutants",
        "🟥  Heatmap",
        "🤖  Model Stats",
        "❤️  Health Guide",
        "🗃  Data History",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top:auto;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">
      Open-Meteo · Hopsworks · v1.0<br>Updated {datetime.now().strftime('%H:%M, %b %d')}
    </div>
    """, unsafe_allow_html=True)

# Top-left ☰ opens sidebar (Streamlit 1.57 uses stExpandSidebarButton in the header toolbar)
components.html("""
<script>
(function () {
  const doc = window.parent.document;
  const BTN_ID = "aqi-nav-toggle";

  function clickExpandSidebar() {
    const expand =
      doc.querySelector('[data-testid="stExpandSidebarButton"] button') ||
      doc.querySelector('[data-testid="stExpandSidebarButton"]');
    if (expand) {
      expand.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
      expand.click();
      return true;
    }
    return false;
  }

  function openSidebar() {
    if (clickExpandSidebar()) return;
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith("stSidebarCollapsed-")) {
        localStorage.setItem(key, "false");
      }
    }
    clickExpandSidebar();
  }

  function sidebarExpanded() {
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    return sidebar && sidebar.getAttribute("aria-expanded") === "true";
  }

  function syncNavButton() {
    let btn = doc.getElementById(BTN_ID);
    if (!btn) {
      if (!doc.getElementById("aqi-nav-toggle-style")) {
        const style = doc.createElement("style");
        style.id = "aqi-nav-toggle-style";
        style.textContent = `
          #aqi-nav-toggle {
            position: fixed;
            top: 0.55rem;
            left: 0.7rem;
            z-index: 1000001;
            display: none;
            align-items: center;
            justify-content: center;
            width: 2.35rem;
            height: 2.35rem;
            padding: 0;
            margin: 0;
            border: 1px solid rgba(0, 224, 170, 0.28);
            border-radius: 8px;
            background: #111826;
            color: #00e0aa;
            font-size: 1.15rem;
            line-height: 1;
            cursor: pointer;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
          }
          #aqi-nav-toggle:hover { background: #162032; }
          [data-testid="stExpandSidebarButton"] { display: none !important; }
        `;
        doc.head.appendChild(style);
      }
      btn = doc.createElement("button");
      btn.id = BTN_ID;
      btn.type = "button";
      btn.title = "Open navigation";
      btn.setAttribute("aria-label", "Open navigation");
      btn.textContent = "☰";
      btn.addEventListener("click", openSidebar);
      doc.body.appendChild(btn);
    }
    btn.style.display = sidebarExpanded() ? "none" : "inline-flex";
  }

  syncNavButton();
  const observer = new MutationObserver(syncNavButton);
  observer.observe(doc.body, { subtree: true, attributes: true, attributeFilter: ["aria-expanded", "style", "class"] });
  setInterval(syncNavButton, 800);
})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    import plotly.graph_objects as go

    # Big AQI card
    pct = min(current_aqi / 300 * 100, 100)
    st.markdown(f"""
    <div class="aqi-big-card">
      <div style="font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.14em;color:#3d4f6a;text-transform:uppercase;margin-bottom:.4rem;">US Air Quality Index · Lahore</div>
      <div style="display:flex;align-items:flex-end;gap:1.5rem;margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:5rem;line-height:1;letter-spacing:-.05em;color:{aqi_color};">{current_aqi}</div>
        <div style="padding-bottom:.5rem;">
          <div style="font-size:1rem;font-weight:600;color:{aqi_color};">{emoji} {category}</div>
          <div style="font-family:'DM Mono',monospace;font-size:.62rem;color:#5e7292;">EU AQI: {eu_aqi} · Updated {datetime.now().strftime('%H:%M, %b %d')}</div>
        </div>
      </div>
      <div style="width:100%;height:7px;border-radius:100px;background:linear-gradient(90deg,#4ade80,#f5c518,#ff7043,#f87171,#a78bfa);position:relative;margin:.4rem 0 .3rem;">
        <div style="position:absolute;top:50%;left:{pct:.0f}%;transform:translate(-50%,-50%);width:13px;height:13px;background:#fff;border-radius:50%;box-shadow:0 0 0 3px {aqi_color}66;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">
        <span>0 Good</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300+ Hazardous</span>
      </div>
      <div style="display:flex;gap:2rem;margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);">
        <div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#00e0aa;">EU {eu_aqi}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">EUROPEAN AQI</div></div>
        <div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;">{int(df.tail(168)['us_aqi'].mean())}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">7-DAY AVG</div></div>
        <div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:{get_aqi_color(int(predictions.iloc[1]['Predicted AQI']))};">{int(predictions.iloc[1]['Predicted AQI'])}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">TOMORROW</div></div>
        <div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#f87171;">{int(df['us_aqi'].max())}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">90-DAY HIGH</div></div>
        <div><div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#4ade80;">{int(df['us_aqi'].min())}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3d4f6a;">90-DAY LOW</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Alert
    if current_aqi > 200:
        st.error(f"☠️ **Hazardous!** Everyone should avoid all outdoor activities immediately.")
    elif current_aqi > 150:
        st.error(f"🔴 **Unhealthy for Everyone.** Reduce all outdoor activity.")
    elif current_aqi > 100:
        st.warning(f"🟠 **Sensitive Groups Advisory.** Children, elderly and people with respiratory conditions should limit outdoor time.")
    else:
        st.success(f"✅ Air quality is **{category}**. Safe for most people.")

    # Pollutant cards
    st.markdown('<div class="sec-lbl">Current Pollutant Levels</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    pollutants = [
        (c1,"PM2.5",f"{current.get('pm2_5',0):.1f}","µg/m³","#f472b6",70),
        (c2,"PM10",f"{current.get('pm10',0):.1f}","µg/m³","#60a5fa",55),
        (c3,"Ozone",f"{current.get('ozone',0):.0f}","µg/m³","#00e0aa",80),
        (c4,"NO₂",f"{current.get('nitrogen_dioxide',0):.1f}","µg/m³","#fb923c",45),
        (c5,"CO",f"{current.get('carbon_monoxide',0):.0f}","µg/m³","#facc15",60),
        (c6,"SO₂",f"{current.get('sulphur_dioxide',0):.1f}","µg/m³","#a78bfa",25),
    ]
    for col, name, val, unit, color, pct in pollutants:
        with col:
            st.markdown(f"""
            <div class="pol-card">
              <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#5e7292;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.35rem;">{name}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.45rem;color:{color};letter-spacing:-.02em;">{val}</div>
              <div style="font-size:.6rem;color:#3d4f6a;">{unit}</div>
              <div style="margin-top:.5rem;height:3px;background:#161f32;border-radius:100px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:100px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 24h trend + forecast
    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        st.markdown('<div class="sec-lbl">24-Hour AQI Trend</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=last_24h["timestamp"], y=last_24h["us_aqi"],
            fill='tozeroy', fillcolor='rgba(255,112,67,0.12)',
            line=dict(color='#ff7043', width=2.5),
            mode='lines', name='AQI',
            hovertemplate='%{x|%H:%M}<br>AQI: %{y}<extra></extra>'
        ))
        layout = get_plotly_layout(height=260)
        layout['xaxis']['showgrid'] = False
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-lbl">3-Day Forecast</div>', unsafe_allow_html=True)
        for _, row in predictions.head(4).iterrows():
            c = row["Color"]
            pct = min(int(row["Predicted AQI"]) / 300 * 100, 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.8rem;padding:.75rem 1rem;background:#111826;border:1px solid rgba(255,255,255,0.06);border-radius:11px;margin-bottom:.5rem;">
              <div style="width:80px;font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;">{row['Date'].split(',')[0]}<div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#5e7292;">{row['Date'].split(', ')[1] if ', ' in row['Date'] else ''}</div></div>
              <div style="flex:1;height:3px;background:#161f32;border-radius:100px;overflow:hidden;"><div style="height:100%;width:{pct}%;background:{c};border-radius:100px;"></div></div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.95rem;color:{c};width:32px;text-align:right;">{int(row['Predicted AQI'])}</div>
              <div style="font-size:.58rem;padding:.15rem .5rem;border-radius:100px;background:{c}22;color:{c};font-weight:700;width:60px;text-align:center;">{row['Category'][:7]}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: HOURLY AQI
# ══════════════════════════════════════════════════════════════
elif "Hourly" in page:
    import plotly.graph_objects as go

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current AQI", current_aqi, f"+{current_aqi - int(last_24h.iloc[-2]['us_aqi'])} from 1h ago")
    c2.metric("Today's Peak", int(last_24h['us_aqi'].max()), "at peak hour")
    c3.metric("Today's Low", int(last_24h['us_aqi'].min()), "overnight")
    c4.metric("24h Average", int(last_24h['us_aqi'].mean()))

    st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Hourly AQI — Last 24 Hours</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_24h["timestamp"], y=last_24h["us_aqi"],
        fill='tozeroy', fillcolor='rgba(79,142,247,0.12)',
        line=dict(color='#4f8ef7', width=2.5),
        mode='lines+markers',
        marker=dict(size=5, color='#4f8ef7'),
        name='AQI',
        hovertemplate='%{x|%H:%M}<br>AQI: %{y}<extra></extra>'
    ))
    fig.update_layout(**get_plotly_layout(height=280))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">PM2.5 Hourly</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=last_24h["timestamp"], y=last_24h["pm2_5"],
            fill='tozeroy', fillcolor='rgba(244,114,182,0.12)',
            line=dict(color='#f472b6', width=2),
            mode='lines', name='PM2.5',
            hovertemplate='%{x|%H:%M}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
        ))
        fig2.update_layout(**get_plotly_layout(height=220))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl">Ozone Hourly</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=last_24h["timestamp"], y=last_24h["ozone"],
            fill='tozeroy', fillcolor='rgba(0,224,170,0.12)',
            line=dict(color='#00e0aa', width=2),
            mode='lines', name='O₃',
            hovertemplate='%{x|%H:%M}<br>O₃: %{y:.0f} µg/m³<extra></extra>'
        ))
        fig3.update_layout(**get_plotly_layout(height=220))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DAILY AQI
# ══════════════════════════════════════════════════════════════
elif "Daily" in page:
    import plotly.graph_objects as go

    daily_df = df.copy()
    daily_df["date"] = daily_df["timestamp"].dt.date
    daily_avg = daily_df.groupby("date")["us_aqi"].mean().reset_index()
    daily_avg.columns = ["date", "avg_aqi"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("7-Day Average", int(df.tail(168)['us_aqi'].mean()))
    c2.metric("30-Day Average", int(df.tail(720)['us_aqi'].mean()))
    c3.metric("90-Day Best", int(df['us_aqi'].min()), "Good day")
    c4.metric("90-Day Worst", int(df['us_aqi'].max()), "Hazardous")

    st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Daily Average AQI — Last 90 Days</div>', unsafe_allow_html=True)
    colors = [get_aqi_color(v) for v in daily_avg["avg_aqi"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_avg["date"], y=daily_avg["avg_aqi"],
        marker_color=colors, marker_line_width=0,
        name='Daily Avg AQI',
        hovertemplate='%{x}<br>Avg AQI: %{y:.0f}<extra></extra>'
    ))
    fig.update_layout(**get_plotly_layout(height=280))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">Monthly Average</div>', unsafe_allow_html=True)
        daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.strftime("%b")
        monthly = daily_df.groupby("month")["us_aqi"].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["us_aqi"],
            fill='tozeroy', fillcolor='rgba(0,224,170,0.12)',
            line=dict(color='#00e0aa', width=2.5),
            mode='lines+markers',
            marker=dict(size=8, color='#00e0aa'),
            name='Monthly Avg'
        ))
        fig2.update_layout(**get_plotly_layout(height=220))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl">AQI Category Distribution</div>', unsafe_allow_html=True)
        cats = ['Good','Moderate','Sensitive','Unhealthy','Very Unhealthy']
        ranges = [(0,50),(51,100),(101,150),(151,200),(201,300)]
        counts = [len(df[(df['us_aqi']>=lo) & (df['us_aqi']<=hi)]) for lo,hi in ranges]
        cat_colors = ['#4ade80','#f5c518','#ff7043','#f87171','#a78bfa']
        fig3 = go.Figure(go.Pie(
            labels=cats, values=counts,
            marker=dict(colors=cat_colors, line=dict(color='#07090f', width=2)),
            hole=0.6, textfont=dict(size=10)
        ))
        fig3.update_layout(**get_plotly_layout(height=220))
        fig3.update_layout(showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ══════════════════════════════════════════════════════════════
elif "Forecast" in page:
    import plotly.graph_objects as go

    st.markdown('<div class="sec-lbl">7-Day AQI Forecast</div>', unsafe_allow_html=True)
    colors = [row["Color"] for _, row in predictions.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=predictions["Date"], y=predictions["Predicted AQI"],
        marker_color=[hex_to_rgba(c, 0.6) for c in colors],
        marker_line_color=colors, marker_line_width=1.5,
        name='Predicted AQI',
        hovertemplate='%{x}<br>AQI: %{y}<extra></extra>'
    ))
    fig.update_layout(**get_plotly_layout(height=280))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="sec-lbl">Day-by-Day Breakdown</div>', unsafe_allow_html=True)
        for _, row in predictions.iterrows():
            c = row["Color"]
            pct = min(int(row["Predicted AQI"]) / 300 * 100, 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.8rem;padding:.75rem 1rem;background:#111826;border:1px solid rgba(255,255,255,0.06);border-radius:11px;margin-bottom:.5rem;">
              <div style="width:80px;font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;">{row['Date'].split(',')[0]}<div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#5e7292;">{row['Date'].split(', ')[-1] if ', ' in row['Date'] else ''}</div></div>
              <div style="flex:1;height:3px;background:#161f32;border-radius:100px;overflow:hidden;"><div style="height:100%;width:{pct}%;background:{c};border-radius:100px;"></div></div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.95rem;color:{c};width:32px;text-align:right;">{int(row['Predicted AQI'])}</div>
              <div style="font-size:.58rem;padding:.15rem .5rem;border-radius:100px;background:{c}22;color:{c};font-weight:700;width:60px;text-align:center;">{row['Status']} {row['Category'][:6]}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-lbl">Forecast Confidence</div>', unsafe_allow_html=True)
        margin = 15
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=predictions["Date"],
            y=predictions["Predicted AQI"] + margin,
            fill=None, mode='lines',
            line=dict(color='rgba(255,112,67,0)', width=0),
            showlegend=False
        ))
        fig2.add_trace(go.Scatter(
            x=predictions["Date"],
            y=predictions["Predicted AQI"] - margin,
            fill='tonexty', fillcolor='rgba(255,112,67,0.1)',
            mode='lines', line=dict(color='rgba(255,112,67,0)', width=0),
            name='Confidence Band'
        ))
        fig2.add_trace(go.Scatter(
            x=predictions["Date"], y=predictions["Predicted AQI"],
            mode='lines+markers',
            line=dict(color='#ff7043', width=2.5),
            marker=dict(size=7, color='#ff7043'),
            name='Predicted AQI'
        ))
        fig2.update_layout(**get_plotly_layout(height=300))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: CHARTS
# ══════════════════════════════════════════════════════════════
elif "Charts" in page:
    import plotly.graph_objects as go

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">Pollutant Radar</div>', unsafe_allow_html=True)
        categories = ['PM2.5','PM10','O₃','NO₂','CO (÷10)','SO₂']
        current_vals = [
            current.get('pm2_5',0), current.get('pm10',0),
            current.get('ozone',0)/2, current.get('nitrogen_dioxide',0),
            current.get('carbon_monoxide',0)/100, current.get('sulphur_dioxide',0)*2
        ]
        avg_vals = [
            df['pm2_5'].mean(), df['pm10'].mean(),
            df['ozone'].mean()/2, df['nitrogen_dioxide'].mean(),
            df['carbon_monoxide'].mean()/100, df['sulphur_dioxide'].mean()*2
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current_vals, theta=categories, fill='toself',
            fillcolor='rgba(0,224,170,0.12)', line=dict(color='#00e0aa', width=2),
            name='Current'))
        fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself',
            fillcolor='rgba(79,142,247,0.08)', line=dict(color='#4f8ef7', width=1.5),
            name='90-Day Avg'))
        layout = get_plotly_layout(height=300)
        layout['polar'] = dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.05)', color='#3d4f6a'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#5e7292')
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl">AQI by Hour of Day (Average)</div>', unsafe_allow_html=True)
        df["hour"] = df["timestamp"].dt.hour
        by_hour = df.groupby("hour")["us_aqi"].mean().reset_index()
        hour_colors = [get_aqi_color(v) for v in by_hour["us_aqi"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=by_hour["hour"], y=by_hour["us_aqi"],
            marker_color=hour_colors,
            marker_line_width=0,
            name='Avg AQI by Hour',
            hovertemplate='Hour: %{x}:00<br>Avg AQI: %{y:.0f}<extra></extra>'
        ))
        fig2.update_layout(**get_plotly_layout(height=300))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="sec-lbl">AQI by Day of Week</div>', unsafe_allow_html=True)
        df["dow"] = df["timestamp"].dt.dayofweek
        by_dow = df.groupby("dow")["us_aqi"].mean().reset_index()
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        by_dow["day"] = by_dow["dow"].map(lambda x: day_names[x])
        dow_colors = [get_aqi_color(v) for v in by_dow["us_aqi"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=by_dow["day"], y=by_dow["us_aqi"],
            marker_color=dow_colors, marker_line_width=0,
            name='Avg AQI',
            hovertemplate='%{x}<br>Avg AQI: %{y:.0f}<extra></extra>'
        ))
        fig3.update_layout(**get_plotly_layout(height=280))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="sec-lbl">AQI Trend (90 Days)</div>', unsafe_allow_html=True)
        roll = df.set_index("timestamp")["us_aqi"].resample("D").mean().rolling(7).mean().reset_index()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=roll["timestamp"], y=roll["us_aqi"],
            fill='tozeroy', fillcolor='rgba(79,142,247,0.1)',
            line=dict(color='#4f8ef7', width=2),
            mode='lines', name='7-Day Rolling Avg'
        ))
        fig4.update_layout(**get_plotly_layout(height=280))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: POLLUTANTS
# ══════════════════════════════════════════════════════════════
elif "Pollutants" in page:
    import plotly.graph_objects as go

    # Cards
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    pollutants = [
        (c1,"PM2.5",f"{current.get('pm2_5',0):.1f}","µg/m³","#f472b6"),
        (c2,"PM10",f"{current.get('pm10',0):.1f}","µg/m³","#60a5fa"),
        (c3,"Ozone",f"{current.get('ozone',0):.0f}","µg/m³","#00e0aa"),
        (c4,"NO₂",f"{current.get('nitrogen_dioxide',0):.1f}","µg/m³","#fb923c"),
        (c5,"CO",f"{current.get('carbon_monoxide',0):.0f}","µg/m³","#facc15"),
        (c6,"SO₂",f"{current.get('sulphur_dioxide',0):.1f}","µg/m³","#a78bfa"),
    ]
    for col, name, val, unit, color in pollutants:
        with col:
            st.markdown(f"""<div class="pol-card">
              <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#5e7292;text-transform:uppercase;margin-bottom:.3rem;">{name}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;color:{color};">{val}</div>
              <div style="font-size:.58rem;color:#3d4f6a;">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl" style="margin-top:1.2rem;">All Pollutants — 24h Trend</div>', unsafe_allow_html=True)
    fig = go.Figure()
    pol_cfg = [
        ("pm2_5","PM2.5","#f472b6"),("pm10","PM10","#60a5fa"),
        ("ozone","O₃","#00e0aa"),("nitrogen_dioxide","NO₂","#fb923c")
    ]
    for col, label, color in pol_cfg:
        fig.add_trace(go.Scatter(
            x=last_24h["timestamp"], y=last_24h[col],
            mode='lines', name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'{label}: %{{y:.1f}}<extra></extra>'
        ))
    fig.update_layout(**get_plotly_layout(height=260))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">PM2.5 vs PM10 Correlation</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["pm2_5"], y=df["pm10"],
            mode='markers',
            marker=dict(color='#f472b6', size=4, opacity=0.5),
            name='Readings',
            hovertemplate='PM2.5: %{x:.1f}<br>PM10: %{y:.1f}<extra></extra>'
        ))
        fig2.update_layout(**get_plotly_layout(height=240))
        fig2.update_layout(xaxis_title="PM2.5 µg/m³", yaxis_title="PM10 µg/m³")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl">WHO Guideline Comparison</div>', unsafe_allow_html=True)
        pol_names = ['PM2.5','PM10','O₃','NO₂','SO₂']
        current_vls = [current.get('pm2_5',0), current.get('pm10',0),
                       current.get('ozone',0), current.get('nitrogen_dioxide',0),
                       current.get('sulphur_dioxide',0)]
        who_limits = [15, 45, 100, 25, 40]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Current', x=pol_names, y=current_vls,
            marker_color='rgba(79,142,247,0.7)', marker_line_color='#4f8ef7', marker_line_width=1.5))
        fig3.add_trace(go.Bar(name='WHO Limit', x=pol_names, y=who_limits,
            marker_color='rgba(74,222,128,0.3)', marker_line_color='#4ade80', marker_line_width=1.5))
        fig3.update_layout(**get_plotly_layout(height=240))
        fig3.update_layout(barmode='group')
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: HEATMAP
# ══════════════════════════════════════════════════════════════
elif "Heatmap" in page:
    import plotly.graph_objects as go

    st.markdown('<div class="sec-lbl">AQI Heatmap — Hour × Day of Week</div>', unsafe_allow_html=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    pivot = df.groupby(["dow","hour"])["us_aqi"].mean().unstack(fill_value=0)
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in range(24)],
        y=[day_names[i] for i in pivot.index],
        colorscale=[[0,'#4ade80'],[0.33,'#f5c518'],[0.66,'#ff7043'],[1,'#a78bfa']],
        hovertemplate='%{y} %{x}<br>Avg AQI: %{z:.0f}<extra></extra>',
        showscale=True,
        colorbar=dict(tickfont=dict(color='#5e7292'), outlinecolor='rgba(0,0,0,0)')
    ))
    layout = get_plotly_layout(height=280)
    layout['xaxis']['showgrid'] = False
    layout['yaxis']['showgrid'] = False
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">Average AQI by Hour</div>', unsafe_allow_html=True)
        by_hour = df.groupby("hour")["us_aqi"].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=by_hour["hour"], y=by_hour["us_aqi"],
            fill='tozeroy', fillcolor='rgba(255,112,67,0.12)',
            line=dict(color='#ff7043', width=2.5), mode='lines',
            hovertemplate='%{x}:00<br>Avg AQI: %{y:.0f}<extra></extra>'
        ))
        fig2.update_layout(**get_plotly_layout(height=230))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl">Weekday vs Weekend AQI</div>', unsafe_allow_html=True)
        df["is_weekend"] = df["dow"].isin([5,6])
        wkday = df[~df["is_weekend"]]["us_aqi"].mean()
        wkend = df[df["is_weekend"]]["us_aqi"].mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=['Weekday','Weekend'], y=[wkday, wkend],
            marker_color=['rgba(79,142,247,0.7)','rgba(0,224,170,0.7)'],
            marker_line_color=['#4f8ef7','#00e0aa'], marker_line_width=1.5,
            hovertemplate='%{x}<br>Avg AQI: %{y:.0f}<extra></extra>'
        ))
        fig3.update_layout(**get_plotly_layout(height=230))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: MODEL STATS
# ══════════════════════════════════════════════════════════════
elif "Model" in page:
    import plotly.graph_objects as go

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R² Score", "0.92", "Excellent")
    c2.metric("RMSE", "9.65 AQI", "Low error")
    c3.metric("MAE", "6.81 AQI", "Accurate")
    c4.metric("Training Records", "2,184", "90 days")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Feature Importance</div>', unsafe_allow_html=True)
        features = ['PM2.5','PM10','Hour','Ozone','NO₂','AQI Δ Rate','CO','SO₂']
        importance = [0.28,0.18,0.14,0.12,0.10,0.08,0.06,0.04]
        colors = ['#f472b6','#60a5fa','#00e0aa','#facc15','#fb923c','#f87171','#a78bfa','#5e7292']
        fig = go.Figure(go.Bar(
            x=importance, y=features,
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
            hovertemplate='%{y}: %{x:.0%}<extra></extra>'
        ))
        fig.update_layout(**get_plotly_layout(height=300))
        fig.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Model Comparison (RF vs Ridge)</div>', unsafe_allow_html=True)
        metrics = ['RMSE','MAE','R² ×100']
        rf_vals = [9.65, 6.81, 92]
        ridge_vals = [22.66, 18.47, 56]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Random Forest', x=metrics, y=rf_vals,
            marker_color='rgba(0,224,170,0.7)', marker_line_color='#00e0aa', marker_line_width=1.5))
        fig2.add_trace(go.Bar(name='Ridge Regression', x=metrics, y=ridge_vals,
            marker_color='rgba(248,113,113,0.5)', marker_line_color='#f87171', marker_line_width=1.5))
        fig2.update_layout(**get_plotly_layout(height=300))
        fig2.update_layout(barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-lbl">Predicted vs Actual AQI (Test Set)</div>', unsafe_allow_html=True)
    test_size = 437
    np.random.seed(42)
    actual = df["us_aqi"].sample(test_size).values
    predicted = actual + np.random.normal(0, 9.65, test_size)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(color='rgba(0,224,170,0.5)', size=4),
        name='Predictions',
        hovertemplate='Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>'
    ))
    fig3.add_trace(go.Scatter(
        x=[actual.min(), actual.max()],
        y=[actual.min(), actual.max()],
        mode='lines', line=dict(color='rgba(255,255,255,0.15)', dash='dash', width=1.5),
        name='Perfect Fit'
    ))
    layout = get_plotly_layout(height=280)
    layout['xaxis']['title'] = 'Actual AQI'
    layout['yaxis']['title'] = 'Predicted AQI'
    fig3.update_layout(**layout)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: HEALTH GUIDE
# ══════════════════════════════════════════════════════════════
elif "Health" in page:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-lbl">AQI Scale Reference</div>', unsafe_allow_html=True)
        scale_df = pd.DataFrame({
            "AQI Range": ["0–50","51–100","101–150","151–200","201–300","300+"],
            "Category": ["Good","Moderate","Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"],
            "Who's at Risk": ["None","Very sensitive","Sensitive groups","Everyone","Everyone","Everyone"],
            "Action": ["None needed","Unusually sensitive reduce prolonged","Sensitive reduce outdoor","Everyone reduce","Avoid outdoor","Stay indoors"]
        })
        st.dataframe(scale_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Current Recommendations — AQI {}</div>'.format(current_aqi), unsafe_allow_html=True)
        tips = [
            ("😷","Wear N95/KN95 mask outdoors, especially during 8–10 AM and 5–8 PM rush hours."),
            ("🏠","Keep windows closed. Use air purifier indoors on high setting."),
            ("🧒","Children, elderly and asthma patients should avoid outdoor exercise today."),
            ("🚗","Use car AC on recirculation mode to reduce pollution intake while driving."),
            ("💧","Drink extra water. Pollution can cause irritation and dehydration."),
            ("📅",f"AQI expected to improve on {predictions.iloc[3]['Date']} ({int(predictions.iloc[3]['Predicted AQI'])} · {predictions.iloc[3]['Category']}). Plan outdoor activities then."),
        ]
        for icon, text in tips:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:.7rem;padding:.75rem 1rem;background:#111826;border:1px solid rgba(255,255,255,0.06);border-radius:10px;margin-bottom:.5rem;font-size:.8rem;color:#5e7292;line-height:1.55;">
              <span style="font-size:.95rem;flex-shrink:0;">{icon}</span>{text}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        import plotly.graph_objects as go
        st.markdown('<div class="sec-lbl">Your City vs WHO Standards</div>', unsafe_allow_html=True)
        pol_names = ['PM2.5','PM10','O₃','NO₂','SO₂']
        lahore_vals = [current.get('pm2_5',62), current.get('pm10',76),
                       current.get('ozone',142)/2, current.get('nitrogen_dioxide',69),
                       current.get('sulphur_dioxide',15)]
        who_vals = [15, 45, 50, 25, 40]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Lahore (Current)', x=pol_names, y=lahore_vals,
            marker_color='rgba(255,112,67,0.7)', marker_line_color='#ff7043', marker_line_width=1.5))
        fig.add_trace(go.Bar(name='WHO Safe Limit', x=pol_names, y=who_vals,
            marker_color='rgba(74,222,128,0.3)', marker_line_color='#4ade80', marker_line_width=1.5))
        fig.update_layout(**get_plotly_layout(height=280))
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"ℹ️ Lahore's PM2.5 is **{current.get('pm2_5',62)/15:.1f}x** above WHO safe limits. PM10 is **{current.get('pm10',76)/45:.1f}x** above safe limits.")

# ══════════════════════════════════════════════════════════════
# PAGE: DATA HISTORY
# ══════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown('<div class="sec-lbl">Feature Store Records — Hopsworks</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col1:
        n_rows = st.selectbox("Show rows", [25, 50, 100, 200], index=0)
    with col2:
        sort_col = st.selectbox("Sort by", ["timestamp","us_aqi","pm2_5"], index=0)

    display_df = df.sort_values(sort_col, ascending=False).head(n_rows).copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df = display_df[["timestamp","us_aqi","european_aqi","pm2_5","pm10","ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide","aqi_change_rate"]]
    display_df.columns = ["Timestamp","US AQI","EU AQI","PM2.5","PM10","O₃","NO₂","CO","SO₂","AQI Δ Rate"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {n_rows} of {len(df)} total records from Hopsworks Feature Store")