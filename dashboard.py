"""
Supply Chain Resilience Dashboard
Benjamin Heo | Lilley Fellowship
Run with: streamlit run dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Resilience AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — full visual overhaul
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
    --bg:       #04060f;
    --bg2:      #080d1c;
    --bg3:      #0c1228;
    --surface:  #101828;
    --surface2: #18243a;
    --border:   #1a2540;
    --border2:  #243050;
    --accent:   #00d4ff;
    --accent2:  #0066ff;
    --accent3:  #7c3aed;
    --gold:     #f59e0b;
    --danger:   #ff3b5c;
    --success:  #00e5a0;
    --text:     #e8edf5;
    --text2:    #8899b4;
    --text3:    #4a5a78;
    --glow:     rgba(0,212,255,0.15);
}

html, body, [class*="css"], .main {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text);
}

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Grid background */
section[data-testid="stMain"] > div {
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--bg2);
    padding: 5px;
    border-radius: 10px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text3);
    border-radius: 7px;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 20px;
    letter-spacing: 0.02em;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text2);
    background: var(--surface);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, var(--surface) 0%, var(--bg3) 100%);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.6;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text3);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.metric-sub {
    font-size: 11px;
    color: var(--text3);
    margin-top: 6px;
}

/* Event cards */
.event-card {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.event-title {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 6px;
}
.event-desc {
    font-size: 13px;
    color: var(--text2);
    line-height: 1.6;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2) 0%, var(--accent) 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    padding: 12px 28px;
    width: 100%;
    letter-spacing: 0.05em;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(0,102,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.4);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

/* Rec cards */
.rec-card {
    background: var(--surface);
    border-left: 3px solid var(--accent2);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: var(--text2);
    line-height: 1.5;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border2), transparent);
    margin: 20px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent2); }

/* Expanders */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 8px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMPORT SIMULATION  (must be in same folder)
# ─────────────────────────────────────────────
try:
    from supply_chain_simulation import (
        simulate, EVENTS, train_rl, Q_TABLE,
        ACTIONS, DAYS, BASE_DEMAND, BASE_CAPACITY
    )
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False
    st.error("⚠️ Could not import supply_chain_simulation.py — make sure it's in the same folder as dashboard.py")

try:
    from ml_risk_model import get_model, EVENT_PROFILES, FEATURES, FEATURE_DESCRIPTIONS, LABEL_COLORS, LABELS
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ─────────────────────────────────────────────
# SESSION STATE  (persists across reruns)
# ─────────────────────────────────────────────
if "rl_trained" not in st.session_state:
    st.session_state.rl_trained = False
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_event_name" not in st.session_state:
    st.session_state.last_event_name = None

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#04060f",
    plot_bgcolor="#101828",
    font=dict(family="JetBrains Mono", color="#8899b4", size=12),
    xaxis=dict(gridcolor="#18243a", linecolor="#18243a", zerolinecolor="#18243a"),
    yaxis=dict(gridcolor="#18243a", linecolor="#18243a", zerolinecolor="#18243a"),
    margin=dict(l=40, r=20, t=40, b=40),
)

POLICY_COLORS = {
    "baseline":     "#ff3b5c",
    "heuristic":    "#f59e0b",
    "rl":           "#00e5a0",
    "optimization": "#00d4ff",
}

POLICY_LABELS = {
    "baseline":     "Baseline",
    "heuristic":    "Heuristic",
    "rl":           "RL Agent",
    "optimization": "LP Optimization",
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='padding:36px 0 24px 0; position:relative;'>"
    "<div style='position:absolute;top:0;left:0;right:0;height:3px;"
    "background:linear-gradient(90deg,#0066ff,#00d4ff,#7c3aed);border-radius:2px;'></div>"
    "<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;margin-top:16px;'>"
    "<div style='width:8px;height:8px;border-radius:50%;background:#00d4ff;box-shadow:0 0 12px #00d4ff;'></div>"
    "<span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#00d4ff;"
    "letter-spacing:0.25em;text-transform:uppercase;'>"
    "Lilley Fellowship &nbsp;&middot;&nbsp; Benjamin Heo &nbsp;&middot;&nbsp; Industrial Engineering + AI"
    "</span></div>"
    "<div style='font-family:Syne,sans-serif;font-size:42px;font-weight:800;"
    "letter-spacing:-0.03em;line-height:1;margin:0 0 4px 0;"
    "background:linear-gradient(135deg,#ffffff 0%,#00d4ff 60%,#0066ff 100%);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>"
    "Supply Chain Resilience</div>"
    "<div style='font-family:Syne,sans-serif;font-size:42px;font-weight:800;"
    "letter-spacing:-0.03em;line-height:1;margin:0 0 20px 0;color:#243050;'>"
    "Intelligence Platform</div>"
    "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;'>"
    "<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#8899b4;'>"
    "AI-driven simulation &middot; geopolitical risk prediction &middot; multi-policy optimization</span>"
    "<span style='background:#101828;border:1px solid #243050;border-radius:20px;"
    "padding:3px 12px;font-size:10px;color:#00d4ff;"
    "font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'>8 HISTORICAL EVENTS</span>"
    "<span style='background:#101828;border:1px solid #243050;border-radius:20px;"
    "padding:3px 12px;font-size:10px;color:#00e5a0;"
    "font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'>ML RISK MODEL</span>"
    "<span style='background:#101828;border:1px solid #243050;border-radius:20px;"
    "padding:3px 12px;font-size:10px;color:#f59e0b;"
    "font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'>RL AGENT</span>"
    "</div></div>",
    unsafe_allow_html=True
)

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 12px 0;border-bottom:1px solid #1a2540;margin-bottom:16px;'>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
        "color:#00d4ff;letter-spacing:0.2em;margin-bottom:6px;'>CONTROL PANEL</div>"
        "<div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700;"
        "color:#e8edf5;'>Simulation</div></div>",
        unsafe_allow_html=True
    )

    if SIM_AVAILABLE:
        event_name = st.selectbox(
            "Historical Event",
            list(EVENTS.keys()),
            index=0,
        )

        st.markdown("**Override Parameters**")
        capacity_override = st.slider(
            "Capacity Factor", 0.1, 1.0,
            float(EVENTS[event_name]["capacity"]), 0.05,
            help="1.0 = full capacity, 0.5 = 50% capacity"
        )
        duration_override = st.slider(
            "Disruption Duration (days)", 10, 300,
            int(EVENTS[event_name]["duration"]), 10
        )
        risk_override = st.slider(
            "Daily Shock Risk", 0.0, 1.0,
            float(EVENTS[event_name]["risk"]), 0.05
        )

        st.divider()

        # RL training button
        if not st.session_state.rl_trained:
            st.warning("RL agent not trained yet.")
            if st.button("🧠 Train RL Agent (500 episodes)"):
                with st.spinner("Training RL agent..."):
                    train_rl(500)
                    st.session_state.rl_trained = True
                st.success("RL agent trained!")
                st.rerun()
        else:
            st.success("✅ RL agent trained")

        st.divider()

        if st.button("▶ Run Simulation"):
            custom_event = EVENTS[event_name].copy()
            custom_event["capacity"] = capacity_override
            custom_event["duration"] = duration_override
            custom_event["risk"]     = risk_override

            results = {}
            policies = ["baseline", "heuristic", "optimization"]
            if st.session_state.rl_trained:
                policies.append("rl")

            with st.spinner("Running simulations..."):
                for p in policies:
                    results[p] = simulate(custom_event, p)

            st.session_state.last_results    = results
            st.session_state.last_event_name = event_name
            st.success("Done! View results in the tabs →")

    else:
        st.error("Simulation module not found.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌐 Network Map",
    "📊 Scenario Runner",
    "🤖 AI Risk Scoring",
    "📈 KPI Dashboard",
    "📚 Case Studies",
    "📡 Live Intelligence",
])

# ══════════════════════════════════════════════
# TAB 1 — NETWORK MAP
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<h3 style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8edf5;margin-bottom:4px">🌐 Global Supply Chain Network</h3>', unsafe_allow_html=True)
    st.markdown("Node color represents disruption risk. Hover over a node for details. Lines show active trade routes.")

    if SIM_AVAILABLE:
        selected_evt = EVENTS.get(event_name if "event_name" in dir() else list(EVENTS.keys())[0], {})
        risk_val = selected_evt.get("risk", 0.3)

        def risk_color(r):
            if r < 0.3:  return "#22c55e"
            if r < 0.6:  return "#f59e0b"
            return "#ef4444"

        # ── Scenario-specific network layouts ──────────────────────
        SCENARIO_NETWORKS = {
            "COVID-19 Pandemic": {
                "description": "Global factory shutdowns centered in Asia; port congestion worldwide.",
                "nodes": [
                    ("Wuhan (Origin)",      {"region": "China",         "risk": 1.0,            "lat": 30.6,  "lon": 114.3}),
                    ("Factory (China)",     {"region": "China",         "risk": risk_val * 1.2, "lat": 31.2,  "lon": 121.5}),
                    ("Supplier (Vietnam)",  {"region": "Vietnam",       "risk": risk_val,       "lat": 21.0,  "lon": 105.8}),
                    ("Port (Shanghai)",     {"region": "China",         "risk": risk_val * 0.9, "lat": 30.6,  "lon": 122.1}),
                    ("Port (LA)",           {"region": "United States", "risk": risk_val * 0.7, "lat": 33.7,  "lon": -118.3}),
                    ("Port (Rotterdam)",    {"region": "Netherlands",   "risk": risk_val * 0.6, "lat": 51.9,  "lon": 4.1}),
                    ("Distributor US",      {"region": "United States", "risk": risk_val * 0.4, "lat": 41.8,  "lon": -87.6}),
                    ("Distributor EU",      {"region": "Germany",       "risk": risk_val * 0.3, "lat": 53.5,  "lon": 10.0}),
                    ("Customer (Global)",   {"region": "Global",        "risk": 0.15,           "lat": 40.7,  "lon": -74.0}),
                ],
                "edges": [
                    ("Wuhan (Origin)",     "Factory (China)"),
                    ("Supplier (Vietnam)", "Factory (China)"),
                    ("Factory (China)",    "Port (Shanghai)"),
                    ("Port (Shanghai)",    "Port (LA)"),
                    ("Port (Shanghai)",    "Port (Rotterdam)"),
                    ("Port (LA)",          "Distributor US"),
                    ("Port (Rotterdam)",   "Distributor EU"),
                    ("Distributor US",     "Customer (Global)"),
                    ("Distributor EU",     "Customer (Global)"),
                ],
            },
            "Suez Canal Blockage": {
                "description": "Single chokepoint in Egypt blocking Asia-Europe trade routes.",
                "nodes": [
                    ("Suez Canal (Blocked)", {"region": "Egypt",         "risk": 1.0,            "lat": 30.5,  "lon": 32.3}),
                    ("Factory (China)",      {"region": "China",         "risk": risk_val * 0.5, "lat": 31.2,  "lon": 121.5}),
                    ("Port (Shanghai)",      {"region": "China",         "risk": risk_val * 0.6, "lat": 30.6,  "lon": 122.1}),
                    ("Port (Singapore)",     {"region": "Singapore",     "risk": risk_val * 0.8, "lat": 1.3,   "lon": 103.8}),
                    ("Alt Route (Cape)",     {"region": "South Africa",  "risk": risk_val * 0.3, "lat": -34.0, "lon": 18.5}),
                    ("Port (Rotterdam)",     {"region": "Netherlands",   "risk": risk_val * 0.9, "lat": 51.9,  "lon": 4.1}),
                    ("Distributor EU",       {"region": "Germany",       "risk": risk_val * 0.7, "lat": 53.5,  "lon": 10.0}),
                    ("Distributor US",       {"region": "United States", "risk": risk_val * 0.4, "lat": 40.7,  "lon": -74.0}),
                    ("Customer (Global)",    {"region": "Global",        "risk": 0.2,            "lat": 48.8,  "lon": 2.3}),
                ],
                "edges": [
                    ("Factory (China)",     "Port (Shanghai)"),
                    ("Port (Shanghai)",     "Port (Singapore)"),
                    ("Port (Singapore)",    "Suez Canal (Blocked)"),
                    ("Suez Canal (Blocked)","Port (Rotterdam)"),
                    ("Port (Singapore)",    "Alt Route (Cape)"),
                    ("Alt Route (Cape)",    "Port (Rotterdam)"),
                    ("Port (Rotterdam)",    "Distributor EU"),
                    ("Port (Rotterdam)",    "Distributor US"),
                    ("Distributor EU",      "Customer (Global)"),
                    ("Distributor US",      "Customer (Global)"),
                ],
            },
            "1973 Oil Embargo": {
                "description": "OPEC nations cutting oil supply to US and Western Europe.",
                "nodes": [
                    ("Saudi Arabia (OPEC)",  {"region": "Saudi Arabia",  "risk": 1.0,            "lat": 24.7,  "lon": 46.7}),
                    ("Iraq (OPEC)",          {"region": "Iraq",          "risk": 0.95,           "lat": 33.3,  "lon": 44.4}),
                    ("Libya (OPEC)",         {"region": "Libya",         "risk": 0.90,           "lat": 32.9,  "lon": 13.2}),
                    ("Strait of Hormuz",     {"region": "Persian Gulf",  "risk": risk_val * 0.9, "lat": 26.6,  "lon": 56.3}),
                    ("Refinery (Houston)",   {"region": "United States", "risk": risk_val * 0.8, "lat": 29.7,  "lon": -95.4}),
                    ("Refinery (Rotterdam)", {"region": "Netherlands",   "risk": risk_val * 0.8, "lat": 51.9,  "lon": 4.1}),
                    ("Factory (US)",         {"region": "United States", "risk": risk_val * 0.6, "lat": 41.8,  "lon": -87.6}),
                    ("Factory (EU)",         {"region": "Germany",       "risk": risk_val * 0.6, "lat": 51.2,  "lon": 6.8}),
                    ("Customer (Global)",    {"region": "Global",        "risk": 0.2,            "lat": 40.7,  "lon": -74.0}),
                ],
                "edges": [
                    ("Saudi Arabia (OPEC)",  "Strait of Hormuz"),
                    ("Iraq (OPEC)",          "Strait of Hormuz"),
                    ("Libya (OPEC)",         "Refinery (Rotterdam)"),
                    ("Strait of Hormuz",     "Refinery (Houston)"),
                    ("Strait of Hormuz",     "Refinery (Rotterdam)"),
                    ("Refinery (Houston)",   "Factory (US)"),
                    ("Refinery (Rotterdam)", "Factory (EU)"),
                    ("Factory (US)",         "Customer (Global)"),
                    ("Factory (EU)",         "Customer (Global)"),
                ],
            },
            "Semiconductor Shortage": {
                "description": "Taiwan and South Korea chip production bottleneck cascading globally.",
                "nodes": [
                    ("TSMC (Taiwan)",        {"region": "Taiwan",        "risk": 1.0,            "lat": 24.8,  "lon": 121.0}),
                    ("Samsung (S. Korea)",   {"region": "South Korea",   "risk": 0.85,           "lat": 37.5,  "lon": 127.0}),
                    ("Fab (Netherlands)",    {"region": "Netherlands",   "risk": risk_val * 0.5, "lat": 51.4,  "lon": 5.5}),
                    ("Auto (Germany)",       {"region": "Germany",       "risk": risk_val * 0.8, "lat": 48.1,  "lon": 11.6}),
                    ("Auto (Michigan)",      {"region": "United States", "risk": risk_val * 0.8, "lat": 42.3,  "lon": -83.0}),
                    ("Electronics (Japan)",  {"region": "Japan",         "risk": risk_val * 0.6, "lat": 35.7,  "lon": 139.7}),
                    ("Distributor US",       {"region": "United States", "risk": risk_val * 0.5, "lat": 37.4,  "lon": -122.0}),
                    ("Distributor EU",       {"region": "France",        "risk": risk_val * 0.4, "lat": 48.8,  "lon": 2.3}),
                    ("Customer (Global)",    {"region": "Global",        "risk": 0.2,            "lat": 40.7,  "lon": -74.0}),
                ],
                "edges": [
                    ("TSMC (Taiwan)",      "Auto (Germany)"),
                    ("TSMC (Taiwan)",      "Auto (Michigan)"),
                    ("TSMC (Taiwan)",      "Electronics (Japan)"),
                    ("Samsung (S. Korea)", "Auto (Germany)"),
                    ("Samsung (S. Korea)", "Electronics (Japan)"),
                    ("Fab (Netherlands)",  "Auto (Germany)"),
                    ("Auto (Germany)",     "Distributor EU"),
                    ("Auto (Michigan)",    "Distributor US"),
                    ("Distributor EU",     "Customer (Global)"),
                    ("Distributor US",     "Customer (Global)"),
                ],
            },
            "Russia-Ukraine Sanctions": {
                "description": "Energy and grain corridors disrupted across Eastern Europe.",
                "nodes": [
                    ("Russia (Energy)",     {"region": "Russia",         "risk": 1.0,            "lat": 55.7,  "lon": 37.6}),
                    ("Ukraine (Grain)",     {"region": "Ukraine",        "risk": 0.95,           "lat": 50.4,  "lon": 30.5}),
                    ("Pipeline (Druzhba)",  {"region": "Poland/Belarus", "risk": risk_val * 0.9, "lat": 52.2,  "lon": 21.0}),
                    ("Black Sea Port",      {"region": "Ukraine",        "risk": risk_val * 0.85,"lat": 46.5,  "lon": 30.7}),
                    ("Refinery (Germany)",  {"region": "Germany",        "risk": risk_val * 0.8, "lat": 51.5,  "lon": 11.9}),
                    ("Grain Hub (Turkey)",  {"region": "Turkey",         "risk": risk_val * 0.5, "lat": 41.0,  "lon": 28.9}),
                    ("Factory (EU)",        {"region": "Germany",        "risk": risk_val * 0.6, "lat": 48.1,  "lon": 11.6}),
                    ("Distributor EU",      {"region": "France",         "risk": risk_val * 0.4, "lat": 48.8,  "lon": 2.3}),
                    ("Customer (Global)",   {"region": "Global",         "risk": 0.2,            "lat": 40.7,  "lon": -74.0}),
                ],
                "edges": [
                    ("Russia (Energy)",    "Pipeline (Druzhba)"),
                    ("Russia (Energy)",    "Refinery (Germany)"),
                    ("Ukraine (Grain)",    "Black Sea Port"),
                    ("Pipeline (Druzhba)", "Refinery (Germany)"),
                    ("Black Sea Port",     "Grain Hub (Turkey)"),
                    ("Grain Hub (Turkey)", "Factory (EU)"),
                    ("Refinery (Germany)", "Factory (EU)"),
                    ("Factory (EU)",       "Distributor EU"),
                    ("Distributor EU",     "Customer (Global)"),
                ],
            },
            "Taiwan Strait Scenario": {
                "description": "Hypothetical conflict blocking Taiwan Strait — worst-case chip supply collapse.",
                "nodes": [
                    ("Taiwan (TSMC)",       {"region": "Taiwan",        "risk": 1.0,            "lat": 23.7,  "lon": 121.0}),
                    ("Taiwan Strait",       {"region": "Taiwan Strait", "risk": 1.0,            "lat": 24.5,  "lon": 119.5}),
                    ("China (PLA)",         {"region": "China",         "risk": 0.95,           "lat": 26.1,  "lon": 119.3}),
                    ("Japan (Alt Fab)",     {"region": "Japan",         "risk": risk_val * 0.7, "lat": 35.7,  "lon": 139.7}),
                    ("S. Korea (Alt Fab)",  {"region": "South Korea",   "risk": risk_val * 0.6, "lat": 37.5,  "lon": 127.0}),
                    ("Port (LA)",           {"region": "United States", "risk": risk_val * 0.8, "lat": 33.7,  "lon": -118.3}),
                    ("Factory (US)",        {"region": "United States", "risk": risk_val * 0.7, "lat": 33.4,  "lon": -112.0}),
                    ("Distributor EU",      {"region": "Germany",       "risk": risk_val * 0.6, "lat": 53.5,  "lon": 10.0}),
                    ("Customer (Global)",   {"region": "Global",        "risk": 0.3,            "lat": 40.7,  "lon": -74.0}),
                ],
                "edges": [
                    ("Taiwan (TSMC)",      "Taiwan Strait"),
                    ("China (PLA)",        "Taiwan Strait"),
                    ("Japan (Alt Fab)",    "Port (LA)"),
                    ("S. Korea (Alt Fab)", "Port (LA)"),
                    ("Port (LA)",          "Factory (US)"),
                    ("Factory (US)",       "Customer (Global)"),
                    ("Japan (Alt Fab)",    "Distributor EU"),
                    ("Distributor EU",     "Customer (Global)"),
                ],
            },
            "Korean War": {
                "description": "Korean War (1950-53): Pacific trade routes disrupted, US strategic materials rationing, UN coalition logistics.",
                "nodes": [
                    ("Korean Peninsula",    {"region": "Korea",         "risk": 1.0,            "lat": 37.5,  "lon": 127.5}),
                    ("Pusan (UN Port)",     {"region": "South Korea",   "risk": risk_val * 0.9, "lat": 35.1,  "lon": 129.0}),
                    ("Japan (Logistics)",   {"region": "Japan",         "risk": risk_val * 0.6, "lat": 34.7,  "lon": 135.5}),
                    ("Port (San Francisco)",{"region": "United States", "risk": risk_val * 0.4, "lat": 37.8,  "lon": -122.4}),
                    ("Factory (Detroit)",   {"region": "United States", "risk": risk_val * 0.5, "lat": 42.3,  "lon": -83.0}),
                    ("UK Supply Hub",       {"region": "United Kingdom", "risk": risk_val * 0.4, "lat": 51.5,  "lon": -0.1}),
                    ("China (PLA Support)", {"region": "China",         "risk": 0.85,           "lat": 41.8,  "lon": 123.4}),
                    ("USSR (Materiel)",     {"region": "Russia",        "risk": 0.80,           "lat": 43.1,  "lon": 131.9}),
                    ("Customer (Allies)",   {"region": "Global",        "risk": 0.2,            "lat": 48.8,  "lon": 2.3}),
                ],
                "edges": [
                    ("Factory (Detroit)",    "Port (San Francisco)"),
                    ("Port (San Francisco)", "Japan (Logistics)"),
                    ("UK Supply Hub",        "Japan (Logistics)"),
                    ("Japan (Logistics)",    "Pusan (UN Port)"),
                    ("Pusan (UN Port)",      "Korean Peninsula"),
                    ("China (PLA Support)",  "Korean Peninsula"),
                    ("USSR (Materiel)",      "China (PLA Support)"),
                    ("Japan (Logistics)",    "Customer (Allies)"),
                ],
            },
            "World War II": {
                "description": "WWII global trade collapse: U-boat warfare in Atlantic, Pacific blockade, full industrial conversion to war production.",
                "nodes": [
                    ("U-Boat Zone (Atlantic)",{"region": "Atlantic Ocean","risk": 1.0,            "lat": 50.0,  "lon": -30.0}),
                    ("Pearl Harbor",          {"region": "Hawaii, USA",   "risk": 0.95,           "lat": 21.4,  "lon": -157.9}),
                    ("Arsenal of Democracy",  {"region": "United States", "risk": 0.30,           "lat": 42.3,  "lon": -83.0}),
                    ("Port (New York)",       {"region": "United States", "risk": risk_val * 0.7, "lat": 40.7,  "lon": -74.0}),
                    ("UK (Receiving)",        {"region": "United Kingdom","risk": risk_val * 0.9, "lat": 51.5,  "lon": -0.1}),
                    ("Murmansk (USSR Aid)",   {"region": "Russia",        "risk": risk_val * 0.95,"lat": 68.9,  "lon": 33.1}),
                    ("Normandy (D-Day)",      {"region": "France",        "risk": 0.90,           "lat": 49.4,  "lon": -0.9}),
                    ("Pacific Theater",       {"region": "Pacific Ocean", "risk": 0.95,           "lat": 10.0,  "lon": 145.0}),
                    ("Axis Europe",           {"region": "Germany",       "risk": 1.0,            "lat": 52.5,  "lon": 13.4}),
                ],
                "edges": [
                    ("Arsenal of Democracy",   "Port (New York)"),
                    ("Port (New York)",        "U-Boat Zone (Atlantic)"),
                    ("U-Boat Zone (Atlantic)", "UK (Receiving)"),
                    ("U-Boat Zone (Atlantic)", "Murmansk (USSR Aid)"),
                    ("UK (Receiving)",         "Normandy (D-Day)"),
                    ("Normandy (D-Day)",       "Axis Europe"),
                    ("Arsenal of Democracy",   "Pearl Harbor"),
                    ("Pearl Harbor",           "Pacific Theater"),
                ],
            },
            "Gaza Conflict & Red Sea Crisis": {
                "description": "Houthi missile attacks forcing container ships off the Red Sea. 15% of global trade rerouted around Africa.",
                "nodes": [
                    ("Houthi Launch Zone",    {"region": "Yemen",         "risk": 1.0,            "lat": 15.4,  "lon": 44.2}),
                    ("Red Sea (Blocked)",     {"region": "Red Sea",       "risk": 1.0,            "lat": 18.0,  "lon": 39.0}),
                    ("Bab-el-Mandeb",         {"region": "Yemen/Djibouti","risk": 0.95,           "lat": 12.6,  "lon": 43.3}),
                    ("Suez Canal",            {"region": "Egypt",         "risk": 0.85,           "lat": 30.5,  "lon": 32.3}),
                    ("Cape of Good Hope",     {"region": "South Africa",  "risk": risk_val * 0.2, "lat": -34.4, "lon": 18.5}),
                    ("Port (Singapore)",      {"region": "Singapore",     "risk": risk_val * 0.4, "lat": 1.3,   "lon": 103.8}),
                    ("Port (Rotterdam)",      {"region": "Netherlands",   "risk": risk_val * 0.6, "lat": 51.9,  "lon": 4.1}),
                    ("Port (Shanghai)",       {"region": "China",         "risk": risk_val * 0.3, "lat": 30.6,  "lon": 122.1}),
                    ("Distributor EU",        {"region": "Germany",       "risk": risk_val * 0.5, "lat": 53.5,  "lon": 10.0}),
                ],
                "edges": [
                    ("Port (Shanghai)",    "Port (Singapore)"),
                    ("Port (Singapore)",   "Bab-el-Mandeb"),
                    ("Bab-el-Mandeb",      "Red Sea (Blocked)"),
                    ("Houthi Launch Zone", "Red Sea (Blocked)"),
                    ("Red Sea (Blocked)",  "Suez Canal"),
                    ("Port (Singapore)",   "Cape of Good Hope"),
                    ("Cape of Good Hope",  "Port (Rotterdam)"),
                    ("Suez Canal",         "Port (Rotterdam)"),
                    ("Port (Rotterdam)",   "Distributor EU"),
                ],
            },
        }

        # Pick network for selected event, fall back to COVID layout
        net = SCENARIO_NETWORKS.get(event_name, SCENARIO_NETWORKS["COVID-19 Pandemic"])
        nodes = net["nodes"]
        edges = net["edges"]

        # Show scenario description
        st.markdown(
            f"<div class='event-card'><div class='event-desc'>{net['description']}</div></div>",
            unsafe_allow_html=True
        )

        node_dict = dict(nodes)

        # ── Emoji icons per transport mode ──────────────────────
        SCENARIO_EMOJI = {
            "COVID-19 Pandemic":        "🚢",
            "Suez Canal Blockage":       "🛳️",
            "1973 Oil Embargo":          "🚛",
            "Semiconductor Shortage":    "✈️",
            "Russia-Ukraine Sanctions":  "🚂",
            "Taiwan Strait Scenario":    "⛵",
        }
        emoji = SCENARIO_EMOJI.get(event_name, "🚢")

        # ── Build animation frames — emoji travels each edge ────
        N_STEPS = 30   # interpolation steps per edge
        frames = []

        # Flatten all edges into one continuous path of waypoints
        all_waypoints = []
        for src, dst in edges:
            s = node_dict[src]; d = node_dict[dst]
            for i in range(N_STEPS):
                t = i / N_STEPS
                all_waypoints.append((
                    s["lat"] + t * (d["lat"] - s["lat"]),
                    s["lon"] + t * (d["lon"] - s["lon"]),
                ))

        # One frame per waypoint
        for wp_lat, wp_lon in all_waypoints:
            frame_data = []
            # Re-add all edge lines (static)
            for src, dst in edges:
                s = node_dict[src]; d = node_dict[dst]
                frame_data.append(go.Scattergeo(
                    lat=[s["lat"], d["lat"]],
                    lon=[s["lon"], d["lon"]],
                    mode="lines",
                    line=dict(width=1.5, color="rgba(56,189,248,0.35)"),
                    hoverinfo="none",
                    showlegend=False,
                ))
            # Re-add all nodes (static)
            for name, attrs in nodes:
                r = min(attrs["risk"], 1.0)
                col = risk_color(r)
                frame_data.append(go.Scattergeo(
                    lat=[attrs["lat"]],
                    lon=[attrs["lon"]],
                    mode="markers+text",
                    marker=dict(size=14, color=col,
                                line=dict(color="#04060f", width=1.5)),
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=9, color="#e8edf5", family="JetBrains Mono"),
                    showlegend=False,
                    hoverinfo="none",
                ))
            # Moving emoji trace
            frame_data.append(go.Scattergeo(
                lat=[wp_lat],
                lon=[wp_lon],
                mode="text",
                text=[emoji],
                textfont=dict(size=20),
                showlegend=False,
                hoverinfo="none",
            ))
            frames.append(go.Frame(data=frame_data))

        # ── Base figure (first frame) ────────────────────────────
        fig_map = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames,
        )

        # If no frames somehow, fall back to static
        if not frames:
            for src, dst in edges:
                s = node_dict[src]; d = node_dict[dst]
                fig_map.add_trace(go.Scattergeo(
                    lat=[s["lat"], d["lat"]],
                    lon=[s["lon"], d["lon"]],
                    mode="lines",
                    line=dict(width=1.5, color="rgba(56,189,248,0.35)"),
                    hoverinfo="none", showlegend=False,
                ))

        # Legend traces (appended after frames so they always show)
        for label, color in [("Low Risk", "#22c55e"), ("Medium Risk", "#f59e0b"), ("High Risk", "#ef4444")]:
            fig_map.add_trace(go.Scattergeo(
                lat=[None], lon=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=label,
                showlegend=True,
            ))

        fig_map.update_geos(
            showland=True,      landcolor="#0c1228",
            showocean=True,     oceancolor="#020408",
            showcoastlines=True, coastlinecolor="#1a2540",
            showlakes=False,
            showcountries=True,  countrycolor="#1a2540",
            showframe=False,
            projection_type="natural earth",
            bgcolor="#04060f",
        )

        fig_map.update_layout(
            paper_bgcolor="#04060f",
            font=dict(family="JetBrains Mono", color="#8899b4", size=12),
            margin=dict(l=0, r=0, t=10, b=0),
            height=520,
            legend=dict(orientation="h", y=-0.02, x=0.3,
                        font=dict(color="#8899b4", size=11),
                        bgcolor="rgba(0,0,0,0)"),
            geo=dict(bgcolor="#04060f"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.08, x=0.1,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=60, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                            mode="immediate",
                        )],
                    ),
                    dict(
                        label="⏸  Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    ),
                ],
                bgcolor="#101828",
                bordercolor="#18243a",
                font=dict(color="#e8edf5", size=12),
            )],
        )

        st.plotly_chart(fig_map, use_container_width=True)


        # ── Animated trade flow widget ─────────────────────────
        SCENARIO_TRANSPORT = {
            "COVID-19 Pandemic":       {"icon": "🚢", "mode": "Container Ship",  "route": "Shanghai → LA → Chicago",          "speed": "slowed 3× by port congestion"},
            "Suez Canal Blockage":     {"icon": "🛳️", "mode": "Cargo Vessel",    "route": "Singapore → Suez → Rotterdam",      "speed": "blocked — rerouting via Cape Horn"},
            "1973 Oil Embargo":        {"icon": "🚛", "mode": "Oil Tanker",      "route": "Persian Gulf → Houston Refinery",    "speed": "embargoed — supply cut by 40%"},
            "Semiconductor Shortage":  {"icon": "✈️", "mode": "Air Freight",    "route": "Taiwan → Japan → Detroit",           "speed": "air-shipped due to chip scarcity"},
            "Russia-Ukraine Sanctions":{"icon": "🚂", "mode": "Rail/Pipeline",  "route": "Moscow → Druzhba Pipeline → Berlin", "speed": "sanctions halting flow"},
            "Taiwan Strait Scenario":  {"icon": "⛵", "mode": "Naval Blockade",  "route": "Taiwan Strait blocked",              "speed": "full naval blockade scenario"},
        }

        transport = SCENARIO_TRANSPORT.get(event_name, list(SCENARIO_TRANSPORT.values())[0])

        st.markdown("""
<style>
@keyframes slide {
    0%   { transform: translateX(-30px); opacity: 0.3; }
    50%  { transform: translateX(calc(100vw - 120px)); opacity: 1; }
    100% { transform: translateX(-30px); opacity: 0.3; }
}
@keyframes slideSlow {
    0%   { transform: translateX(-30px); opacity: 0.2; }
    50%  { transform: translateX(calc(100vw - 120px)); opacity: 0.7; }
    100% { transform: translateX(-30px); opacity: 0.2; }
}
.transport-bar {
    background: linear-gradient(90deg, #101828, #0f172a, #101828);
    border: 1px solid #18243a;
    border-radius: 12px;
    padding: 18px 24px;
    margin: 12px 0 20px 0;
    overflow: hidden;
    position: relative;
    height: 80px;
}
.transport-icon {
    font-size: 28px;
    position: absolute;
    top: 24px;
    animation: slide 4s ease-in-out infinite;
    filter: drop-shadow(0 0 6px rgba(56,189,248,0.6));
}
.transport-icon-2 {
    font-size: 22px;
    position: absolute;
    top: 28px;
    animation: slideSlow 4s ease-in-out infinite;
    animation-delay: 1.5s;
    filter: drop-shadow(0 0 4px rgba(56,189,248,0.4));
}
.transport-label {
    position: absolute;
    bottom: 10px;
    left: 24px;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    color: #4a5a78;
    letter-spacing: 0.08em;
}
.transport-route {
    position: absolute;
    top: 10px;
    right: 20px;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    color: #00d4ff;
    text-align: right;
}
.route-line {
    position: absolute;
    top: 50%;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #0066ff44, #00d4ff22, #0066ff44, transparent);
}
</style>
""", unsafe_allow_html=True)

        risk_pct = int(selected_evt.get("risk", 0.3) * 100)
        disrupted = risk_pct >= 50

        st.markdown(f"""
<div class="transport-bar">
    <div class="route-line"></div>
    <div class="transport-icon">{transport["icon"]}</div>
    <div class="transport-icon-2">{transport["icon"]}</div>
    <div class="transport-route">
        {transport["route"]}<br>
        <span style="color:#{'ef4444' if disrupted else '22c55e'}; font-size:10px;">
            ● {transport["speed"]}
        </span>
    </div>
    <div class="transport-label">{transport["mode"]}  ·  Risk Level: {risk_pct}%</div>
</div>
""", unsafe_allow_html=True)

        # ── What does this map mean? explainer ─────────────────
        st.markdown("#### 📖 How to Read This Map")

        EXPLAINERS = {
            "COVID-19 Pandemic": {
                "what": "This map shows the global manufacturing and shipping network during the COVID-19 pandemic. The red node at Wuhan marks the disruption origin. Chinese factories lost ~45% capacity while global ports experienced severe congestion.",
                "nodes": "**Red nodes** = severely disrupted (factories shut down, ports congested). **Yellow nodes** = partially disrupted. **Green nodes** = relatively stable but experiencing ripple effects.",
                "lines": "**Blue lines** = active trade routes. Thicker = higher volume. During COVID, all lines slowed dramatically as ships queued outside ports for weeks.",
                "takeaway": "🔑 Key insight: Over-reliance on a single manufacturing hub (China) created catastrophic single-point-of-failure risk. Companies with diversified supplier bases recovered 2–3× faster.",
            },
            "Suez Canal Blockage": {
                "what": "The Ever Given container ship blocked the Suez Canal for 6 days in March 2021, halting ~12% of global trade. This map shows how a single geographic chokepoint can paralyze worldwide logistics.",
                "nodes": "**Red node** at Suez = total blockage point. Yellow nodes at Rotterdam and Singapore show downstream congestion. The Cape of Good Hope node shows the emergency alternate route.",
                "lines": "The line through Suez turns red (blocked). The alternate route via South Africa adds 9–12 extra days of transit time and significantly higher fuel costs.",
                "takeaway": "🔑 Key insight: The canal handles 30% of global container traffic. One ship caused .6B/day in halted trade — illustrating how geographic concentration creates extreme fragility.",
            },
            "1973 Oil Embargo": {
                "what": "OPEC nations cut oil exports to the US and Western Europe in response to support for Israel in the Yom Kippur War. This map shows the energy supply chain from oil fields through the Strait of Hormuz to refineries.",
                "nodes": "**Red nodes** = OPEC supplier nations enforcing the embargo. The Strait of Hormuz is a critical chokepoint — 20% of global oil passes through it. Houston and Rotterdam refineries show downstream impact.",
                "lines": "Lines from Saudi Arabia and Iraq represent oil tanker routes. These were partially blocked or redirected during the embargo, causing oil prices to rise 300%.",
                "takeaway": "🔑 Key insight: Energy is the hidden input in every supply chain. When fuel costs spike, ALL logistics costs spike — this was the first major lesson in supply chain systemic risk.",
            },
            "Semiconductor Shortage": {
                "what": "Taiwan produces 92% of the world's most advanced chips. A combination of COVID demand surge, factory fires, and drought (chips need huge amounts of water) created a global shortage lasting 3+ years.",
                "nodes": "**TSMC (Taiwan)** and **Samsung (S. Korea)** are shown in red — these two companies supply chips to virtually every electronics and automotive manufacturer on the planet.",
                "lines": "Lines connect chip fabs directly to auto plants (Germany, Michigan) and electronics factories (Japan). When chips stopped flowing, car production halted even though cars themselves were in high demand.",
                "takeaway": "🔑 Key insight: Just-in-time manufacturing with zero safety stock is catastrophically fragile for critical inputs. The auto industry lost 7.7M vehicles of production from a  chip.",
            },
            "Russia-Ukraine Sanctions": {
                "what": "Russia's invasion of Ukraine in 2022 triggered sweeping sanctions that disrupted two critical supply chains simultaneously: Russian energy to Europe, and Ukrainian grain to the world.",
                "nodes": "**Russia** (energy) and **Ukraine** (grain) are shown in red. The Druzhba pipeline and Black Sea port nodes show the disrupted transit routes. Germany and France show downstream factory impacts.",
                "lines": "Pipeline lines represent natural gas flows that were cut off. Black Sea shipping lanes were blocked by naval conflict, trapping Ukrainian grain and spiking global food prices.",
                "takeaway": "🔑 Key insight: Geopolitical risk concentrates over time — Europe's dependence on Russian gas grew for decades. Diversification is only cheap BEFORE a crisis.",
            },
            "Taiwan Strait Scenario": {
                "what": "A hypothetical military conflict over Taiwan would be the most severe supply chain shock in modern history. Taiwan produces chips that power everything from phones to missiles. There is no short-term substitute.",
                "nodes": "**Taiwan (TSMC)** shown in deep red — loss of this single node would eliminate 92% of advanced chip production globally. China (PLA) shown as the threat node. Japan and South Korea as partial alternatives.",
                "lines": "The Taiwan Strait line is shown as blocked. Alternative routes via Japan and South Korea can only replace ~20% of Taiwan's capacity in the short term.",
                "takeaway": "🔑 Key insight: .6 trillion in annual electronics production depends on Taiwan. This is why chip reshoring (CHIPS Act) and allied production capacity are now treated as national security issues.",
            },
        }

        expl = EXPLAINERS.get(event_name, list(EXPLAINERS.values())[0])

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown(
                f"<div class='event-card' style='border-left-color:#00d4ff'>"
                f"<div class='event-title'>🌐 What This Map Shows</div>"
                f"<div class='event-desc'>{expl['what']}</div>"
                f"</div>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='event-card' style='border-left-color:#22c55e'>"
                f"<div class='event-title'>⭕ What the Nodes Mean</div>"
                f"<div class='event-desc'>{expl['nodes']}</div>"
                f"</div>", unsafe_allow_html=True
            )
        with col_e2:
            st.markdown(
                f"<div class='event-card' style='border-left-color:#f59e0b'>"
                f"<div class='event-title'>📍 What the Lines Mean</div>"
                f"<div class='event-desc'>{expl['lines']}</div>"
                f"</div>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='event-card' style='border-left-color:#a855f7'>"
                f"<div class='event-title'>{expl['takeaway']}</div>"
                f"</div>", unsafe_allow_html=True
            )

        # Node risk table
        st.markdown("#### Node Risk Summary")
        node_df = pd.DataFrame([
            {
                "Node": name,
                "Region": attrs["region"],
                "Coordinates": f"{attrs['lat']}°N, {attrs['lon']}°E",
                "Risk Score": f"{min(attrs['risk'],1.0):.0%}",
                "Status": "🔴 High" if attrs["risk"] >= 0.6
                          else ("🟡 Medium" if attrs["risk"] >= 0.3 else "🟢 Low")
            }
            for name, attrs in nodes
        ])
        st.dataframe(node_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 2 — SCENARIO RUNNER
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<h3 style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8edf5;margin-bottom:4px">📊 Scenario Runner</h3>', unsafe_allow_html=True)

    if st.session_state.last_results is None:
        st.info("👈 Configure parameters in the sidebar and click **▶ Run Simulation** to see results.")
    else:
        results      = st.session_state.last_results
        evt_name     = st.session_state.last_event_name

        st.markdown(f"**Event:** {evt_name}")
        if SIM_AVAILABLE and evt_name in EVENTS:
            st.markdown(
                f"<div class='event-card'>"
                f"<div class='event-desc'>{EVENTS[evt_name]['description']}</div>"
                f"</div>", unsafe_allow_html=True
            )

        # Backlog chart
        st.markdown("#### Daily Backlog by Policy")
        fig_bl = go.Figure()
        for policy, res in results.items():
            fig_bl.add_trace(go.Scatter(
                y=res["backlog"],
                mode="lines",
                name=POLICY_LABELS.get(policy, policy),
                line=dict(color=POLICY_COLORS.get(policy, "#fff"), width=2),
                hovertemplate="%{y:.0f} units<extra>" + POLICY_LABELS.get(policy, policy) + "</extra>",
            ))
        fig_bl.add_vline(
            x=EVENTS[evt_name]["duration"] if evt_name in EVENTS else 0,
            line_dash="dot", line_color="#4a5a78",
            annotation_text="Disruption ends",
            annotation_font_color="#4a5a78",
        )
        fig_bl.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=380, xaxis_title="Day", yaxis_title="Backlog (units)")
        st.plotly_chart(fig_bl, use_container_width=True)

        # Inventory chart
        st.markdown("#### Daily Inventory by Policy")
        fig_inv = go.Figure()
        for policy, res in results.items():
            fig_inv.add_trace(go.Scatter(
                y=res["inventory"],
                mode="lines",
                name=POLICY_LABELS.get(policy, policy),
                line=dict(color=POLICY_COLORS.get(policy, "#fff"), width=2),
            ))
        fig_inv.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=320, xaxis_title="Day", yaxis_title="Inventory (units)")
        st.plotly_chart(fig_inv, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — ML RISK MODEL
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 ML Geopolitical Risk Prediction Model")
    st.markdown("Gradient Boosting ensemble trained on 24 historical disruption observations. Predicts supply chain risk from 8 geopolitical indicators.")

    if not ML_AVAILABLE:
        st.error("ml_risk_model.py not found. Make sure it's in the same folder as dashboard.py.")
    else:
        ml = get_model()
        cur_event = st.session_state.last_event_name or list(EVENTS.keys())[0]

        # ── Model accuracy row ───────────────────────────────────
        st.markdown("#### Model Performance (5-Fold Cross-Validation)")
        cv_results = ml.cross_validate()
        cv_cols = st.columns(3)
        model_colors = {"Gradient Boosting": "#00d4ff", "Random Forest": "#22c55e", "Logistic Regression": "#f59e0b"}
        for i, (mname, scores) in enumerate(cv_results.items()):
            with cv_cols[i]:
                st.markdown(
                    f"<div class='metric-card' style='border-color:{model_colors[mname]}44'>"
                    f"<div class='metric-label'>{mname}</div>"
                    f"<div class='metric-value' style='color:{model_colors[mname]}'>{scores['mean_accuracy']:.0%}</div>"
                    f"<div class='metric-sub'>± {scores['std']:.0%} accuracy</div>"
                    f"</div>", unsafe_allow_html=True
                )

        st.divider()

        # ── Event prediction ─────────────────────────────────────
        st.markdown(f"#### Risk Assessment: {cur_event}")
        result = ml.predict_event(cur_event)

        if result:
            col_a, col_b, col_c = st.columns([1, 1, 2])

            lcolor = LABEL_COLORS.get(result["risk_label"], "#4a5a78")
            with col_a:
                st.markdown(
                    f"<div class='metric-card' style='border-color:{lcolor}'>"
                    f"<div class='metric-label'>Risk Score</div>"
                    f"<div class='metric-value' style='color:{lcolor}'>{result['risk_score']:.3f}</div>"
                    f"<div class='metric-sub'>{result['risk_label']} Risk</div>"
                    f"</div>", unsafe_allow_html=True
                )

            with col_b:
                st.markdown("<div class='metric-card'><div class='metric-label'>Probability Breakdown</div>", unsafe_allow_html=True)
                for lbl, prob in result["probabilities"].items():
                    lc = LABEL_COLORS.get(lbl, "#4a5a78")
                    bar_w = int(prob * 100)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                        f"<span style='font-family:JetBrains Mono;font-size:11px;color:{lc};width:60px'>{lbl}</span>"
                        f"<div style='flex:1;background:#18243a;border-radius:4px;height:10px'>"
                        f"<div style='width:{bar_w}%;background:{lc};height:10px;border-radius:4px'></div></div>"
                        f"<span style='font-size:11px;color:#4a5a78'>{prob:.0%}</span>"
                        f"</div>", unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_c:
                st.markdown(
                    f"<div class='event-card'>"
                    f"<div class='event-title'>Model Narrative</div>"
                    f"<div class='event-desc' style='margin-top:6px'>{result['narrative']}</div>"
                    f"</div>", unsafe_allow_html=True
                )

            # Top drivers
            st.markdown("#### Top Risk Drivers")
            feat_cols = st.columns(3)
            for i, (feat, imp) in enumerate(result["top_features"]):
                with feat_cols[i]:
                    bar_pct = int(imp * 300)
                    desc = FEATURE_DESCRIPTIONS.get(feat, "")
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-label'>Driver #{i+1}</div>"
                        f"<div style='font-family:JetBrains Mono;font-size:14px;color:#00d4ff;margin:6px 0'>{feat.replace('_',' ').title()}</div>"
                        f"<div style='background:#18243a;border-radius:4px;height:8px;margin:6px 0'>"
                        f"<div style='width:{min(bar_pct,100)}%;background:#00d4ff;height:8px;border-radius:4px'></div></div>"
                        f"<div style='font-size:11px;color:#4a5a78'>{desc}</div>"
                        f"</div>", unsafe_allow_html=True
                    )

            # Recommendations
            st.markdown("#### 📋 AI Recommendations")
            for rec in result["recommendations"]:
                st.markdown(
                    f"<div style='background:#101828;border-left:3px solid #0066ff;"
                    f"padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0;"
                    f"font-size:13px;color:#cbd5e1'>{rec}</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Risk timeline chart ──────────────────────────────────
        st.markdown("#### ML Risk Score Timeline (300 Days)")
        timeline = ml.predict_timeline(cur_event, days=300)
        tl_scores = [t["risk_score"] for t in timeline]

        fig_ml = go.Figure()
        fig_ml.add_hrect(y0=0.66, y1=1.0,  fillcolor="rgba(239,68,68,0.07)",  line_width=0,
                         annotation_text="CRITICAL", annotation_font_color="#ef4444", annotation_position="top left")
        fig_ml.add_hrect(y0=0.33, y1=0.66, fillcolor="rgba(245,158,11,0.06)", line_width=0,
                         annotation_text="HIGH",     annotation_font_color="#f59e0b", annotation_position="top left")
        fig_ml.add_hrect(y0=0.1,  y1=0.33, fillcolor="rgba(34,197,94,0.04)",  line_width=0,
                         annotation_text="MEDIUM",   annotation_font_color="#22c55e", annotation_position="top left")
        fig_ml.add_trace(go.Scatter(
            y=tl_scores, mode="lines", fill="tozeroy",
            fillcolor="rgba(30,64,175,0.10)",
            line=dict(color="#00d4ff", width=2), name="ML Risk Score",
        ))
        fig_ml.add_hline(y=0.66, line_dash="dash", line_color="rgba(239,68,68,0.4)")
        fig_ml.add_hline(y=0.33, line_dash="dash", line_color="rgba(245,158,11,0.4)")
        fig_ml.update_layout(
            paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
            font=dict(family="JetBrains Mono", color="#8899b4", size=12),
            margin=dict(l=40, r=20, t=40, b=40), height=320,
            xaxis_title="Day",
            yaxis=dict(title="Risk Score", range=[0, 1.05], gridcolor="#18243a"),
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # ── All-events comparison radar ──────────────────────────
        st.markdown("#### All-Events Risk Comparison")
        all_events_ml = list(EVENT_PROFILES.keys())
        scores_all = []
        labels_all = []
        colors_all = []
        for en in all_events_ml:
            r = ml.predict_event(en)
            if r:
                scores_all.append(r["risk_score"])
                labels_all.append(r["risk_label"])
                colors_all.append(LABEL_COLORS.get(r["risk_label"], "#4a5a78"))

        fig_bar_all = go.Figure(go.Bar(
            x=all_events_ml,
            y=scores_all,
            marker_color=colors_all,
            text=[f"{s:.2f} — {l}" for s, l in zip(scores_all, labels_all)],
            textposition="outside",
            textfont=dict(color="#8899b4", size=10),
        ))
        fig_bar_all.update_layout(
            paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
            font=dict(family="JetBrains Mono", color="#8899b4", size=11),
            margin=dict(l=40, r=20, t=20, b=100), height=380,
            xaxis=dict(tickangle=-30, gridcolor="#18243a"),
            yaxis=dict(range=[0, 1.15], gridcolor="#18243a", title="Risk Score"),
        )
        st.plotly_chart(fig_bar_all, use_container_width=True)

        # RL action distribution (kept here)
        if st.session_state.rl_trained and Q_TABLE:
            st.markdown("#### RL Agent Action Distribution")
            action_counts = defaultdict(int)
            for (state_key, action), q_val in Q_TABLE.items():
                if q_val > 0:
                    action_counts[action] += 1
            if action_counts:
                fig_actions = go.Figure(go.Bar(
                    x=list(action_counts.keys()),
                    y=list(action_counts.values()),
                    marker_color=["#00d4ff","#22c55e","#f59e0b","#a855f7","#4a5a78"],
                    text=list(action_counts.values()),
                    textposition="outside",
                    textfont=dict(color="#8899b4"),
                ))
                fig_actions.update_layout(
                    paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                    font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                    margin=dict(l=40, r=20, t=40, b=40), height=280,
                    xaxis_title="Action", yaxis_title="Q-Table Entries",
                )
                st.plotly_chart(fig_actions, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — KPI DASHBOARD
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<h3 style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8edf5;margin-bottom:4px">📈 KPI Dashboard</h3>', unsafe_allow_html=True)

    if st.session_state.last_results is None:
        st.info("👈 Run a simulation first to see KPIs.")
    else:
        results  = st.session_state.last_results
        policies = list(results.keys())

        # Top metric cards
        cols = st.columns(len(policies))
        for i, policy in enumerate(policies):
            res = results[policy]
            with cols[i]:
                color = POLICY_COLORS.get(policy, "#fff")
                st.markdown(
                    f"<div class='metric-card' style='border-color:{color}44'>"
                    f"<div class='metric-label'>{POLICY_LABELS.get(policy, policy)}</div>"
                    f"<div class='metric-value' style='color:{color}'>{res['avg_backlog']:.0f}</div>"
                    f"<div class='metric-sub'>avg backlog (units)</div>"
                    f"</div>", unsafe_allow_html=True
                )

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            # Service level bar chart
            st.markdown("#### Service Level %")
            fig_svc = go.Figure(go.Bar(
                x=[POLICY_LABELS.get(p, p) for p in policies],
                y=[results[p]["service_level_pct"] for p in policies],
                marker_color=[POLICY_COLORS.get(p, "#fff") for p in policies],
                text=[f"{results[p]['service_level_pct']:.1f}%" for p in policies],
                textposition="outside",
                textfont=dict(color="#e8edf5"),
            ))
            fig_svc.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=300, yaxis=dict(range=[0, 110], gridcolor="#18243a"))
            st.plotly_chart(fig_svc, use_container_width=True)

        with c2:
            # Total cost bar chart
            st.markdown("#### Total Cost ($)")
            fig_cost = go.Figure(go.Bar(
                x=[POLICY_LABELS.get(p, p) for p in policies],
                y=[results[p]["total_cost"] for p in policies],
                marker_color=[POLICY_COLORS.get(p, "#fff") for p in policies],
                text=[f"${results[p]['total_cost']:,.0f}" for p in policies],
                textposition="outside",
                textfont=dict(color="#e8edf5"),
            ))
            fig_cost.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=300)
            st.plotly_chart(fig_cost, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### Max Backlog (units)")
            fig_max = go.Figure(go.Bar(
                x=[POLICY_LABELS.get(p, p) for p in policies],
                y=[results[p]["max_backlog"] for p in policies],
                marker_color=[POLICY_COLORS.get(p, "#fff") for p in policies],
                text=[f"{results[p]['max_backlog']:.0f}" for p in policies],
                textposition="outside",
                textfont=dict(color="#e8edf5"),
            ))
            fig_max.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=300)
            st.plotly_chart(fig_max, use_container_width=True)

        with c4:
            st.markdown("#### Recovery Day")
            rec_vals = [results[p]["recovery_day"] or DAYS for p in policies]
            fig_rec = go.Figure(go.Bar(
                x=[POLICY_LABELS.get(p, p) for p in policies],
                y=rec_vals,
                marker_color=[POLICY_COLORS.get(p, "#fff") for p in policies],
                text=[str(v) if v < DAYS else "Not recovered" for v in rec_vals],
                textposition="outside",
                textfont=dict(color="#e8edf5"),
            ))
            fig_rec.update_layout(paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=12),
                margin=dict(l=40, r=20, t=40, b=40), height=300, yaxis_title="Day")
            st.plotly_chart(fig_rec, use_container_width=True)

        # Summary table
        st.markdown("#### Full KPI Table")
        kpi_rows = []
        for policy in policies:
            r = results[policy]
            kpi_rows.append({
                "Policy":           POLICY_LABELS.get(policy, policy),
                "Avg Backlog":      f"{r['avg_backlog']:.1f}",
                "Max Backlog":      f"{r['max_backlog']:.1f}",
                "Service Level %":  f"{r['service_level_pct']:.1f}%",
                "Recovery Day":     str(r['recovery_day']) if r['recovery_day'] else "N/A",
                "Inventory Turns":  f"{r['inventory_turns']:.2f}",
                "Total Cost ($)":   f"${r['total_cost']:,.0f}",
                "Transport Cost":   f"${r['transport_cost']:,.0f}",
            })
        st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

        # ── AI-Generated KPI Explanation ────────────────────────
        st.divider()
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#00d4ff;letter-spacing:0.2em;text-transform:uppercase;"
            "margin-bottom:12px;'>RESULTS INTERPRETATION</div>",
            unsafe_allow_html=True
        )

        # Find best policy per KPI
        best_svc   = max(policies, key=lambda p: results[p]["service_level_pct"])
        best_cost  = min(policies, key=lambda p: results[p]["total_cost"])
        best_bl    = min(policies, key=lambda p: results[p]["avg_backlog"])
        best_rec   = min(policies, key=lambda p: results[p]["recovery_day"] or DAYS)
        worst_bl   = max(policies, key=lambda p: results[p]["avg_backlog"])

        bl_improvement = (
            (results[worst_bl]["avg_backlog"] - results[best_bl]["avg_backlog"])
            / max(results[worst_bl]["avg_backlog"], 1) * 100
        )
        svc_gap = (
            results[best_svc]["service_level_pct"]
            - results[worst_bl]["service_level_pct"]
        )
        rec_best = results[best_rec]["recovery_day"] or DAYS
        rec_worst = max(results[p]["recovery_day"] or DAYS for p in policies)

        evt = st.session_state.last_event_name or "this event"
        evt_params = EVENTS.get(evt, {})
        severity = "severe" if evt_params.get("capacity", 1) < 0.5 else "moderate"
        duration_label = "long-duration" if evt_params.get("duration", 0) > 150 else "short-duration"

        explanation_blocks = [
            {
                "title": f"Winner: {POLICY_LABELS.get(best_bl, best_bl)} had the lowest average backlog",
                "color": POLICY_COLORS.get(best_bl, "#00d4ff"),
                "body": (
                    f"Across the {duration_label} {severity} disruption of {evt}, "
                    f"{POLICY_LABELS.get(best_bl, best_bl)} achieved an average backlog of "
                    f"{results[best_bl]['avg_backlog']:.0f} units — "
                    f"{bl_improvement:.0f}% lower than the worst-performing policy. "
                    f"This means fewer unfulfilled orders per day, directly translating to higher customer satisfaction and lower penalty costs."
                ),
            },
            {
                "title": f"Service Level: {POLICY_LABELS.get(best_svc, best_svc)} filled {results[best_svc]['service_level_pct']:.1f}% of demand",
                "color": POLICY_COLORS.get(best_svc, "#00e5a0"),
                "body": (
                    f"Service level measures what percentage of daily demand was fulfilled on time. "
                    f"A {svc_gap:.1f} percentage point gap between best and worst policies shows how much "
                    f"adaptive decision-making matters under {severity} disruption. "
                    f"In real supply chains, every 1% drop in service level can represent millions in lost revenue and damaged customer relationships."
                ),
            },
            {
                "title": f"Recovery: {POLICY_LABELS.get(best_rec, best_rec)} recovered on Day {rec_best}",
                "color": POLICY_COLORS.get(best_rec, "#f59e0b"),
                "body": (
                    f"Recovery day is when backlog first drops below 10 units — effectively back to normal operations. "
                    f"The fastest policy ({POLICY_LABELS.get(best_rec, best_rec)}) recovered {rec_worst - rec_best} days "
                    f"earlier than the slowest. In a real disruption like {evt}, that gap represents weeks of lost productivity "
                    f"and compounding costs for the slower-responding firm."
                ),
            },
            {
                "title": f"Cost: {POLICY_LABELS.get(best_cost, best_cost)} was cheapest at ${results[best_cost]['total_cost']:,.0f}",
                "color": POLICY_COLORS.get(best_cost, "#a855f7"),
                "body": (
                    f"Total cost includes production, holding, backlog penalty, and transportation costs. "
                    f"Note that the cheapest policy is not always the best — a policy might minimize cost by "
                    f"accepting more backlog. The ideal strategy balances cost against service level. "
                    f"Compare {POLICY_LABELS.get(best_cost, best_cost)}'s cost vs. its service level above to judge true efficiency."
                ),
            },
        ]

        for block in explanation_blocks:
            st.markdown(
                f"<div class='event-card' style='border-left-color:{block["color"]};margin-bottom:10px'>"
                f"<div class='event-title' style='color:{block["color"]}'>{block["title"]}</div>"
                f"<div class='event-desc'>{block["body"]}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Bottom takeaway
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#101828,#0c1228);"
            f"border:1px solid #243050;border-radius:12px;padding:18px 20px;margin-top:8px;'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            f"color:#00d4ff;letter-spacing:0.15em;margin-bottom:8px;'>KEY TAKEAWAY</div>"
            f"<div style='font-family:Syne,sans-serif;font-size:15px;font-weight:600;color:#e8edf5;'>"
            f"Under {evt}, the {POLICY_LABELS.get(best_bl, best_bl)} policy outperformed baseline "
            f"by reducing average backlog {bl_improvement:.0f}% while recovering "
            f"{rec_worst - rec_best} days faster — demonstrating that AI-driven supply chain "
            f"optimization creates measurable, quantifiable resilience gains over reactive management."
            f"</div></div>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════
# TAB 5 — HISTORICAL CASE STUDIES
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<h3 style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8edf5;margin-bottom:4px">📚 Historical Case Studies</h3>', unsafe_allow_html=True)
    st.markdown("Each event is modeled using calibrated historical parameters. Click an event to see its profile.")

    if SIM_AVAILABLE:
        CASE_STUDY_DETAILS = {
            "COVID-19 Pandemic": {
                "period": "2020–2022",
                "impact": "Global factory shutdowns reduced capacity by ~45%. Demand surged 80% in essential goods. Port congestion tripled lead times.",
                "key_lesson": "Diversified supplier networks and safety stock buffers were the primary resilience factors.",
                "sources": "McKinsey Global Institute (2021), WTO Trade Statistics",
                "color": "#ff3b5c",
            },
            "Suez Canal Blockage": {
                "period": "March 2021 (6 days)",
                "impact": "$9.6B/day in trade halted. ~369 ships delayed. European-Asian routes extended by 9–12 days.",
                "key_lesson": "Single-point-of-failure bottlenecks create outsized risk. Route redundancy is critical.",
                "sources": "Lloyd's List, IMF Working Paper 2021",
                "color": "#f59e0b",
            },
            "1973 Oil Embargo": {
                "period": "October 1973 – March 1974",
                "impact": "Oil prices rose 300%. Transportation costs surged. Demand for non-essential goods collapsed by 25%.",
                "key_lesson": "Energy dependency is a systemic supply chain risk; fuel hedging strategies proved essential.",
                "sources": "OECD Economic Outlook, US DOE Historical Records",
                "color": "#a855f7",
            },
            "Semiconductor Shortage": {
                "period": "2020–2023",
                "impact": "Auto production cut by 7.7M vehicles. Electronics lead times exceeded 52 weeks in some categories.",
                "key_lesson": "Just-in-time manufacturing without strategic stockpiles is catastrophically fragile for critical inputs.",
                "sources": "SEMI Industry Association, IHS Markit",
                "color": "#00e5a0",
            },
            "Russia-Ukraine Sanctions": {
                "period": "February 2022–present",
                "impact": "Ukraine supplies 12% of global wheat. Russia supplies 40% of EU natural gas. Energy and food inflation cascaded globally.",
                "key_lesson": "Geopolitical concentration risk requires scenario planning and alternative sourcing before crises occur.",
                "sources": "UNCTAD 2022, World Bank Commodity Markets",
                "color": "#00d4ff",
            },
            "Taiwan Strait Scenario": {
                "period": "Hypothetical",
                "impact": "Taiwan produces 92% of the world's most advanced chips. A conflict could eliminate $1.6T in electronics production annually.",
                "key_lesson": "Strategic reshoring and allied production capacity are the only meaningful hedges against this tail risk.",
                "sources": "CSIS Taiwan Strait Report, Semiconductor Industry Association",
                "color": "#fb923c",
            },
            "Korean War": {
                "period": "June 1950 – July 1953",
                "impact": "Korean industrial output destroyed. US triggered strategic materials rationing under the Defense Production Act. Pacific shipping lanes militarized. The UN coalition assembled the most complex multinational logistics operation since WWII, supplying 16 nations across the Pacific.",
                "key_lesson": "Military conflict can instantly nationalize supply chains. Governments will commandeer private logistics capacity and materials — corporations must plan for force majeure at nation-state scale.",
                "sources": "US Army Center of Military History, Defense Production Act Records, UN Command Logistics Reports",
                "color": "#7c3aed",
            },
            "World War II": {
                "period": "1939–1945 (peak disruption 1942–1945)",
                "impact": "The most severe supply chain collapse in history. U-boat warfare sank 2,779 Allied merchant ships in the Atlantic. The Pacific was entirely closed to civilian trade. All major industrial economies converted 40–60% of GDP to war production. Civilian goods effectively ceased. The Allied logistics miracle — supplying D-Day with 150,000 troops in 24 hours and simultaneously sustaining the Pacific theater — remains the greatest supply chain achievement ever.",
                "key_lesson": "At maximum disruption, supply chains don't bend — they break entirely and are replaced by command economies. Resilience planning must include scenarios where markets stop functioning.",
                "sources": "US War Production Board, Allied Logistics Command Records, OECD Historical Statistics, Overy — Why the Allies Won (1995)",
                "color": "#8b5cf6",
            },
            "Gaza Conflict & Red Sea Crisis": {
                "period": "October 2023–present",
                "impact": "Houthi attacks on Red Sea shipping forced 90%+ of container ships to reroute around Africa's Cape of Good Hope, adding 10–14 days and $1M+ per voyage. Red Sea normally carries 15% of global trade. Suez Canal revenues dropped 60%. Container freight rates tripled on Asia-Europe routes. Israeli port operations disrupted. Regional energy markets spiked.",
                "key_lesson": "Asymmetric warfare (a non-state actor with missiles) can shut down a global trade artery more effectively than state actors. Insurance and rerouting costs are now a permanent risk premium on Middle East-adjacent trade.",
                "sources": "UNCTAD Red Sea Crisis Report (2024), Freightos Baltic Index, Lloyd's Market Association, IMF Regional Economic Outlook",
                "color": "#ef4444",
            },
        }

        for evt_name, details in CASE_STUDY_DETAILS.items():
            evt_params = EVENTS.get(evt_name, {})
            with st.expander(f"{'🔴' if details['color']=='#ef4444' else '🟡' if details['color']=='#f59e0b' else '🔵'}  {evt_name}  ·  {details['period']}"):
                col_a, col_b = st.columns([3, 2])

                with col_a:
                    st.markdown(f"**Impact:** {details['impact']}")
                    st.markdown(f"**Key Lesson:** {details['key_lesson']}")
                    st.markdown(f"*Sources: {details['sources']}*")

                with col_b:
                    st.markdown("**Simulation Parameters**")
                    param_df = pd.DataFrame([
                        {"Parameter": "Capacity Factor",    "Value": f"{evt_params.get('capacity','-'):.0%}"},
                        {"Parameter": "Demand Multiplier",  "Value": f"{evt_params.get('demand_multiplier','-'):.1f}×"},
                        {"Parameter": "Lead Time (days)",   "Value": str(evt_params.get('lead_time', '-'))},
                        {"Parameter": "Daily Shock Risk",   "Value": f"{evt_params.get('risk','-'):.0%}"},
                        {"Parameter": "Duration (days)",    "Value": str(evt_params.get('duration', '-'))},
                    ])
                    st.dataframe(param_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 6 — LIVE INTELLIGENCE FEED
# ══════════════════════════════════════════════
with tab6:
    import urllib.request
    import xml.etree.ElementTree as ET
    import time as time_mod
    import hashlib
    import re

    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
        "color:#e8edf5;margin-bottom:4px'>📡 Live Supply Chain Intelligence</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:#8899b4;"
        "margin-bottom:20px'>Real-time news from Reuters, BBC, NPR, Al Jazeera, The Guardian & NYT. "
        "Auto-refreshes every 5 minutes. Risk-tagged by AI keyword matching.</div>",
        unsafe_allow_html=True
    )

    # ── Risk keyword classifier ───────────────────────────────────
    RISK_KEYWORDS = {
        "CRITICAL": ["war","invasion","nuclear","blockade","sanctions","embargo","attack",
                     "missile","conflict","bombing","siege","coup","assassination","explosion"],
        "HIGH":     ["shortage","disruption","port closure","strike","tariff","inflation",
                     "crisis","recession","drought","flood","hurricane","earthquake","protest",
                     "shutdown","bankruptcy","default","spike","surge","plunge","collapse"],
        "MEDIUM":   ["tension","supply chain","freight","shipping","delay","slowdown","trade",
                     "commodity","oil","energy","weather","rate","price","cost","deal","merger",
                     "earnings","gdp","unemployment","factory","production","output"],
        "LOW":      ["agreement","recovery","growth","investment","diplomacy","ceasefire",
                     "export","import","forecast","outlook","report","data","survey"],
    }
    RISK_COLORS = {"CRITICAL":"#ff3b5c","HIGH":"#f59e0b","MEDIUM":"#00d4ff","LOW":"#00e5a0"}
    RISK_ICONS  = {"CRITICAL":"🚨","HIGH":"⚠️","MEDIUM":"📌","LOW":"✅"}

    def classify_risk(text):
        text_lower = text.lower()
        for level in ["CRITICAL","HIGH","MEDIUM","LOW"]:
            if any(kw in text_lower for kw in RISK_KEYWORDS[level]):
                return level
        return "MEDIUM"  # default to MEDIUM not LOW so more articles show

    # ── "What this means" implication generator ───────────────────
    # Rule-based, not an LLM call: matches an article's text against topic
    # keyword groups (ordered by specificity) and returns a short supply-chain
    # implication sentence plus a suggested watch-item. Same heuristic spirit
    # as the live disruption forecast above — a directional read, not a
    # certainty.
    IMPLICATION_TOPICS = [
        ("chokepoint", ["suez","hormuz","red sea","strait","panama canal","malacca",
                        "bab-el-mandeb","canal blocked","port closure","port closed"],
         "A key shipping chokepoint or port is implicated — expect rerouting, longer transit times, "
         "and higher freight/insurance costs on any route through this corridor."),
        ("conflict", ["war","invasion","missile","airstrike","bombing","offensive","troops",
                      "military strike","attack","siege"],
         "Active military conflict raises the odds of route closures, cargo insurance spikes, and "
         "possible commandeering of transport capacity in the affected region."),
        ("sanctions", ["sanctions","embargo","export ban","export controls","blacklist"],
         "New or tightened sanctions can cut off suppliers overnight — firms sourcing from or "
         "trading through this corridor should audit exposure and line up compliant alternatives."),
        ("energy", ["oil","opec","pipeline","lng","crude","refinery","gas prices","energy crisis"],
         "Energy market moves here tend to pass through quickly to freight fuel surcharges and "
         "production costs across manufacturing-heavy supply chains."),
        ("semiconductor", ["semiconductor","chip shortage","chipmaker","taiwan chips","tsmc"],
         "Chip-sector disruption hits nearly every downstream industry (auto, electronics, defense) "
         "given how concentrated advanced semiconductor production is."),
        ("tariff", ["tariff","trade war","import duty","customs duty"],
         "New tariffs change landed costs immediately — sourcing and pricing models should be "
         "re-run for affected product lines."),
        ("labor", ["strike","port workers","dockworkers","walkout","union","labor dispute"],
         "A labor action at ports or in freight/logistics can create backlogs that take weeks "
         "to clear even after it's resolved."),
        ("weather", ["hurricane","earthquake","flood","drought","typhoon","wildfire","storm"],
         "Natural disasters disrupt regional production and transport infrastructure — impact "
         "depends heavily on whether key factories, ports, or rail lines sit in the affected area."),
        ("trade_concentration", ["single supplier","sole source","dependent on china","monopoly"],
         "Concentrated sourcing means a localized event can still cause an outsized, hard-to-hedge "
         "disruption for buyers with no qualified second source."),
        ("macro", ["recession","gdp","inflation","interest rate","central bank","default","bankruptcy"],
         "Macroeconomic stress tends to show up in supply chains as demand volatility, "
         "tighter trade credit, and higher counterparty risk."),
    ]

    def generate_implication(text, risk_level):
        text_lower = text.lower()
        for _, kws, sentence in IMPLICATION_TOPICS:
            if any(kw in text_lower for kw in kws):
                return sentence
        # No specific topic matched — fall back to a risk-level-appropriate generic note
        if risk_level in ("CRITICAL", "HIGH"):
            return ("Flagged as elevated risk by keyword pattern, but no specific supply-chain "
                    "driver was identified — worth a manual read before acting on it.")
        return ("No clear supply-chain driver detected in this headline — likely background "
                "noise rather than an actionable disruption signal.")

    # ── Reliable RSS feeds (tested, no auth needed) ───────────────
    # NOTE: Reuters discontinued its public feeds.reuters.com RSS feeds in 2020 —
    # replaced below with CBS News World and Fox News World, which are confirmed live.
    FEEDS = [
        {"name":"CBS World",       "url":"https://www.cbsnews.com/latest/rss/world",             "color":"#ff8c00"},
        {"name":"Fox World",       "url":"https://moxie.foxnews.com/google-publisher/world.xml", "color":"#ff6600"},
        {"name":"BBC World",       "url":"https://feeds.bbci.co.uk/news/world/rss.xml",          "color":"#cc0000"},
        {"name":"BBC Business",    "url":"https://feeds.bbci.co.uk/news/business/rss.xml",       "color":"#ee1111"},
        {"name":"NPR World",       "url":"https://feeds.npr.org/1004/rss.xml",                   "color":"#4a90d9"},
        {"name":"NPR Economy",     "url":"https://feeds.npr.org/1006/rss.xml",                   "color":"#2563eb"},
        {"name":"Al Jazeera",      "url":"https://www.aljazeera.com/xml/rss/all.xml",            "color":"#f5a623"},
        {"name":"The Guardian",    "url":"https://www.theguardian.com/world/rss",                "color":"#00b2ff"},
        {"name":"NYT World",       "url":"https://rss.nytimes.com/services/xml/rss/nyt/World.xml","color":"#444444"},
    ]

    # Very broad — almost any article will match at least one term
    SUPPLY_CHAIN_TERMS = [
        # Direct SC terms
        "supply chain","logistics","freight","shipping","cargo","port","warehouse",
        "import","export","trade","tariff","sanction","embargo","customs",
        # Commodities & energy
        "oil","gas","fuel","energy","commodity","grain","wheat","corn","steel","copper",
        "semiconductor","chip","battery","lithium","rare earth",
        # Economic signals
        "inflation","recession","gdp","interest rate","federal reserve","central bank",
        "dollar","currency","exchange rate","stock market","earnings","debt",
        # Geopolitical hotspots
        "ukraine","russia","china","taiwan","iran","north korea","israel","gaza",
        "red sea","suez","hormuz","south china sea","nato","g7","g20","opec",
        "middle east","asia pacific","europe","africa",
        # Disruption events
        "war","conflict","strike","protest","shortage","disruption","delay","crisis",
        "hurricane","earthquake","flood","drought","pandemic","covid",
        # Companies & sectors
        "manufacturing","factory","production","automotive","technology","pharmaceutical",
        "boeing","apple","tesla","amazon","walmart","maersk",
    ]

    import ssl
    try:
        import certifi
        SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # Fall back to the system default; if that still fails to verify
        # (common on some Mac Python installs missing root certs), use an
        # unverified context so the feed still loads.
        try:
            SSL_CONTEXT = ssl.create_default_context()
        except Exception:
            SSL_CONTEXT = ssl._create_unverified_context()

    def fetch_feed(feed):
        """Fetch and parse RSS, return ALL articles (no filter — show everything)."""
        articles = []
        last_error = None
        try:
            req = urllib.request.Request(
                feed["url"],
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "application/rss+xml, application/xml, text/xml, */*",
                }
            )
            try:
                resp = urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT)
            except ssl.SSLCertVerificationError:
                # Retry once without cert verification rather than failing silently
                resp = urllib.request.urlopen(req, timeout=10, context=ssl._create_unverified_context())
            with resp:
                raw = resp.read()
            # Strip namespaces for easier parsing
            raw_str = raw.decode("utf-8", errors="replace")
            raw_str = re.sub(r' xmlns[^"]*"[^"]*"', '', raw_str)
            root = ET.fromstring(raw_str.encode("utf-8"))
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items[:15]:
                def get_text(tags):
                    for tag in tags:
                        el = item.find(tag)
                        if el is not None:
                            return (el.text or "").strip()
                        # try with namespace
                        el = item.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
                        if el is not None:
                            return (el.text or "").strip()
                    return ""
                title = get_text(["title"])
                desc  = get_text(["description","summary","content"])
                link  = get_text(["link"])
                date  = get_text(["pubDate","published","updated"])
                if not title:
                    continue
                # Clean HTML tags from desc
                desc = re.sub(r"<[^>]+>", "", desc)
                desc = desc[:200] + "…" if len(desc) > 200 else desc
                combined = (title + " " + desc).lower()
                # Only filter out truly irrelevant articles (sports, entertainment)
                skip_terms = ["nfl","nba","oscar","grammy","celebrity","movie","album",
                              "recipe","fashion","sport","football","basketball","baseball",
                              "soccer","tennis","golf","olympics","emmy","grammy"]
                if any(s in combined for s in skip_terms):
                    continue
                risk = classify_risk(combined)
                articles.append({
                    "source":       feed["name"],
                    "source_color": feed["color"],
                    "title":        title,
                    "desc":         desc,
                    "link":         link,
                    "date":         date[:25] if date else "",
                    "risk":         risk,
                    "relevant":     any(t in combined for t in SUPPLY_CHAIN_TERMS),
                })
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        return articles, last_error

    # ── Fetch with caching (5-min TTL) ───────────────────────────
    cache_key = "live_feed_cache"
    cache_time_key = "live_feed_time"

    now_ts = time_mod.time()
    last_fetch = st.session_state.get(cache_time_key, 0)
    CACHE_TTL = 300  # 5 minutes

    col_refresh, col_status = st.columns([1, 3])
    with col_refresh:
        force_refresh = st.button("🔄 Refresh Now")

    if force_refresh or (now_ts - last_fetch > CACHE_TTL) or cache_key not in st.session_state:
        with st.spinner("Scanning live intelligence feeds..."):
            all_articles = []
            feed_errors = {}
            for feed in FEEDS:
                arts, err = fetch_feed(feed)
                all_articles.extend(arts)
                if err:
                    feed_errors[feed["name"]] = err
            all_articles.sort(key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x["risk"]))
            st.session_state[cache_key] = all_articles
            st.session_state[cache_time_key] = now_ts
            st.session_state["live_feed_errors"] = feed_errors
    else:
        all_articles = st.session_state[cache_key]
        feed_errors = st.session_state.get("live_feed_errors", {})

    with col_status:
        elapsed = int(now_ts - last_fetch)
        next_refresh = max(0, CACHE_TTL - elapsed)
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:#4a5a78;padding-top:12px;'>"
            f"Last updated {elapsed}s ago &nbsp;·&nbsp; "
            f"Next auto-refresh in {next_refresh}s &nbsp;·&nbsp; "
            f"{len(all_articles)} supply-chain articles found</div>",
            unsafe_allow_html=True
        )

    # ── Risk summary bar ──────────────────────────────────────────
    if all_articles:
        from collections import Counter
        risk_counts = Counter(a["risk"] for a in all_articles)
        summary_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px;'>"
        for level in ["CRITICAL","HIGH","MEDIUM","LOW"]:
            cnt = risk_counts.get(level, 0)
            col = RISK_COLORS[level]
            ico = RISK_ICONS[level]
            summary_html += (
                f"<div style='background:#101828;border:1px solid {col}44;"
                f"border-radius:10px;padding:12px 20px;min-width:100px;text-align:center;'>"
                f"<div style='font-size:20px;margin-bottom:4px;'>{ico}</div>"
                f"<div style='font-family:Syne,sans-serif;font-size:22px;"
                f"font-weight:800;color:{col};'>{cnt}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                f"color:#4a5a78;letter-spacing:0.1em;'>{level}</div>"
                f"</div>"
            )
        summary_html += "</div>"
        st.markdown(summary_html, unsafe_allow_html=True)

    # ── LIVE DISRUPTION FORECAST ──────────────────────────────────
    # Bridges the live news feed into the ML risk model: keyword density
    # across today's articles is mapped to the same 8-feature vector used
    # for historical events, then scored by the trained GeopoliticalRiskModel.
    # This is a heuristic proxy for NLP feature extraction, not a validated
    # NLP pipeline — treat the output as a directional signal, not ground truth.
    if all_articles and ML_AVAILABLE:
        st.divider()
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:18px;font-weight:800;"
            "color:#e8edf5;margin-bottom:4px'>🔮 Live Disruption Forecast</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#8899b4;"
            "margin-bottom:16px'>Today's headlines run through the same ML risk model used for "
            "historical case studies. Feature values are estimated from keyword density across "
            "all fetched articles — an approximation, not a trained NLP extractor.</div>",
            unsafe_allow_html=True
        )

        FEATURE_KEYWORDS = {
            "conflict_intensity":    ["war","invasion","military","missile","attack","bombing",
                                      "conflict","strike","offensive","troops","siege"],
            "trade_concentration":   ["single supplier","sole source","dependent on china",
                                      "taiwan chips","monopoly","concentration","one country"],
            "energy_dependency":     ["oil","gas","energy","fuel","opec","pipeline","lng",
                                      "crude","refinery","electricity"],
            "chokepoint_exposure":   ["suez","hormuz","red sea","strait","canal","malacca",
                                      "shipping lane","port closure","maritime"],
            "sanctions_active":      ["sanctions","embargo","export ban","export controls",
                                      "blacklist","tariff"],
            "gdp_shock":             ["recession","gdp","contraction","downturn","default",
                                      "bankruptcy","collapse","plunge"],
            "lead_time_variability": ["delay","backlog","congestion","shortage","disruption",
                                      "bottleneck","slowdown"],
            "political_instability": ["coup","protest","unrest","election","instability",
                                       "regime","government collapse","assassination"],
        }

        corpus = [(a["title"] + " " + a["desc"]).lower() for a in all_articles]
        n_docs = len(corpus)

        live_features = {}
        for feat, kws in FEATURE_KEYWORDS.items():
            hits = sum(1 for doc in corpus if any(kw in doc for kw in kws))
            # density (0-1), softened so a handful of hits doesn't max out the score
            live_features[feat] = min(1.0, (hits / n_docs) * 2.2) if n_docs else 0.0

        model = get_model()
        live_result = model.predict(live_features)

        lc1, lc2 = st.columns([1, 2])
        with lc1:
            rlabel = live_result["risk_label"]
            rcol = LABEL_COLORS.get(rlabel, "#8899b4")
            st.markdown(
                f"<div style='background:#101828;border:1px solid {rcol}55;border-radius:14px;"
                f"padding:20px;text-align:center;'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a5a78;"
                f"letter-spacing:0.15em;margin-bottom:8px;'>CURRENT NEWS-DERIVED RISK</div>"
                f"<div style='font-family:Syne,sans-serif;font-size:36px;font-weight:800;"
                f"color:{rcol};'>{live_result['risk_score']:.2f}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;"
                f"color:{rcol};margin-top:4px;'>{rlabel.upper()}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with lc2:
            st.markdown("**Top drivers in today's news**")
            for feat, imp in live_result["top_features"]:
                st.markdown(
                    f"- **{feat.replace('_',' ').title()}** "
                    f"({live_features[feat]:.2f} estimated intensity, "
                    f"{imp:.0%} model weight) — {FEATURE_DESCRIPTIONS.get(feat,'')}"
                )
            st.markdown("**Suggested actions**")
            for rec in live_result["recommendations"]:
                st.markdown(f"- {rec}")

        with st.expander("How this estimate is calculated"):
            st.markdown(
                "For each of the 8 model features, we count how many fetched articles contain "
                "any of that feature's associated keywords, then convert that share into a 0–1 "
                "intensity score. That 8-value vector is fed into the same trained Gradient "
                "Boosting / Random Forest / Logistic Regression ensemble used for historical "
                "event scoring (see AI Risk Scoring tab). Limitations: keyword matching can't "
                "capture sentiment, sarcasm, or article importance, and a single dramatic "
                "headline is weighted the same as a routine one. Treat this as an early-warning "
                "signal to investigate further, not a precise forecast."
            )

    # ── Filter controls ───────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        filter_risk = st.multiselect(
            "Risk Level",
            ["CRITICAL","HIGH","MEDIUM","LOW"],
            default=["CRITICAL","HIGH","MEDIUM","LOW"]
        )
    with fc2:
        filter_source = st.multiselect(
            "Source",
            [f["name"] for f in FEEDS],
            default=[f["name"] for f in FEEDS]
        )
    with fc3:
        sc_only = st.toggle("Supply Chain Relevant Only", value=False)

    filtered = [
        a for a in all_articles
        if a["risk"] in filter_risk
        and a["source"] in filter_source
        and (not sc_only or a.get("relevant", True))
    ]

    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
        f"color:#4a5a78;margin-bottom:16px;'>"
        f"Showing {len(filtered)} of {len(all_articles)} articles fetched</div>",
        unsafe_allow_html=True
    )

    # ── Article cards ─────────────────────────────────────────────
    if not filtered:
        st.info("No articles match your filters, or feeds are temporarily unavailable. Try refreshing.")
        if feed_errors:
            with st.expander(f"⚠️ {len(feed_errors)} of {len(FEEDS)} feeds failed — show details"):
                for name, err in feed_errors.items():
                    st.markdown(f"**{name}**: `{err}`")
                st.caption(
                    "Common fixes: if you see a certificate error, run "
                    "'Install Certificates.command' from your Python.app folder (Mac) or "
                    "`pip install --upgrade certifi`. If you see a connection/timeout error, "
                    "check your network/firewall/VPN — some networks block outbound RSS requests."
                )
    else:
        for art in filtered:
            rc  = RISK_COLORS[art["risk"]]
            ri  = RISK_ICONS[art["risk"]]
            sc  = art["source_color"]
            implication = generate_implication(art["title"] + " " + art["desc"], art["risk"])

            st.markdown(
                f"<a href='{art['link']}' target='_blank' style='text-decoration:none;'>"
                f"<div style='background:linear-gradient(135deg,#101828,#0c1228);"
                f"border:1px solid #1a2540;border-left:4px solid {rc};"
                f"border-radius:0 14px 14px 0;padding:16px 20px;margin-bottom:10px;"
                f"transition:all 0.2s;cursor:pointer;"
                f"'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
                f"<span style='font-size:16px;'>{ri}</span>"
                f"<span style='background:{rc}22;border:1px solid {rc}44;border-radius:6px;"
                f"padding:2px 8px;font-family:JetBrains Mono,monospace;font-size:9px;"
                f"color:{rc};letter-spacing:0.1em;font-weight:700;'>{art['risk']}</span>"
                f"<span style='background:{sc}22;border:1px solid {sc}44;border-radius:6px;"
                f"padding:2px 8px;font-family:JetBrains Mono,monospace;font-size:9px;"
                f"color:{sc};letter-spacing:0.1em;'>{art['source']}</span>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
                f"color:#4a5a78;margin-left:auto;'>{art['date']}</span>"
                f"</div>"
                f"<div style='font-family:Syne,sans-serif;font-size:15px;font-weight:700;"
                f"color:#e8edf5;margin-bottom:6px;line-height:1.3;'>{art['title']}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                f"color:#8899b4;line-height:1.5;'>{art['desc']}</div>"
                f"<div style='margin-top:10px;padding-top:10px;border-top:1px solid #1a2540;"
                f"font-family:JetBrains Mono,monospace;font-size:11px;color:{rc};"
                f"line-height:1.5;'>"
                f"<span style='opacity:0.8;letter-spacing:0.05em;'>🧭 WHAT THIS MEANS:</span> "
                f"<span style='color:#a8b4cc;'>{implication}</span>"
                f"</div>"
                f"</div></a>",
                unsafe_allow_html=True
            )

    # ── Live risk radar chart ─────────────────────────────────────
    if all_articles:
        st.divider()
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#00d4ff;letter-spacing:0.2em;text-transform:uppercase;"
            "margin-bottom:12px;'>LIVE RISK DISTRIBUTION</div>",
            unsafe_allow_html=True
        )

        risk_counts = Counter(a["risk"] for a in all_articles)
        source_counts = Counter(a["source"] for a in all_articles)

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_risk_dist = go.Figure(go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                marker=dict(colors=[RISK_COLORS.get(k,"#fff") for k in risk_counts.keys()],
                            line=dict(color="#04060f", width=2)),
                textfont=dict(family="JetBrains Mono", size=11, color="#e8edf5"),
                hole=0.55,
            ))
            fig_risk_dist.update_layout(
                paper_bgcolor="#04060f", plot_bgcolor="#04060f",
                font=dict(family="JetBrains Mono", color="#8899b4"),
                margin=dict(l=10,r=10,t=30,b=10), height=280,
                title=dict(text="By Risk Level", font=dict(color="#8899b4",size=11)),
                legend=dict(font=dict(color="#8899b4",size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_risk_dist, use_container_width=True)

        with ch2:
            fig_src_dist = go.Figure(go.Bar(
                x=list(source_counts.keys()),
                y=list(source_counts.values()),
                marker_color=[f["color"] for f in FEEDS if f["name"] in source_counts],
                text=list(source_counts.values()),
                textposition="outside",
                textfont=dict(color="#8899b4"),
            ))
            fig_src_dist.update_layout(
                paper_bgcolor="#04060f", plot_bgcolor="#080d1c",
                font=dict(family="JetBrains Mono", color="#8899b4", size=11),
                margin=dict(l=20,r=20,t=30,b=20), height=280,
                title=dict(text="Articles by Source", font=dict(color="#8899b4",size=11)),
                yaxis=dict(gridcolor="#18243a"),
                xaxis=dict(gridcolor="#18243a"),
            )
            st.plotly_chart(fig_src_dist, use_container_width=True)

        # ── Trending keywords ─────────────────────────────────────
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#00d4ff;letter-spacing:0.2em;text-transform:uppercase;"
            "margin-bottom:12px;margin-top:8px;'>TRENDING KEYWORDS</div>",
            unsafe_allow_html=True
        )
        from collections import Counter as Ctr
        import re
        all_text = " ".join(a["title"] + " " + a["desc"] for a in all_articles).lower()
        stopwords = {"the","a","an","in","of","to","and","is","are","for","on","at",
                     "by","with","from","that","this","it","be","as","was","were",
                     "has","have","its","or","but","not","s","will","said","also",
                     "after","over","more","new","he","she","they","his","her","their"}
        words = [w for w in re.findall(r"[a-z]{4,}", all_text) if w not in stopwords]
        top_words = Ctr(words).most_common(20)

        keyword_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;'>"
        max_count = top_words[0][1] if top_words else 1
        for word, count in top_words:
            intensity = count / max_count
            if intensity > 0.7:   col = "#ff3b5c"
            elif intensity > 0.4: col = "#f59e0b"
            else:                 col = "#00d4ff"
            size = int(10 + intensity * 8)
            keyword_html += (
                f"<span style='background:{col}18;border:1px solid {col}44;"
                f"border-radius:20px;padding:4px 12px;"
                f"font-family:JetBrains Mono,monospace;font-size:{size}px;"
                f"color:{col};'>{word} <span style='opacity:0.5;font-size:9px;'>×{count}</span></span>"
            )
        keyword_html += "</div>"
        st.markdown(keyword_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; padding:16px 0 8px 0;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                color:#1a2540; letter-spacing:0.15em; text-transform:uppercase;">
        Lilley Fellowship &nbsp;·&nbsp; Benjamin Heo &nbsp;·&nbsp;
        Supply Chain Resilience Intelligence Platform &nbsp;·&nbsp;
        Industrial Engineering + AI + Geopolitical Risk
    </div>
</div>
""", unsafe_allow_html=True)
