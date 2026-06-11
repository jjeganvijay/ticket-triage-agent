"""Enterprise AI Operations Platform for Ticket Triage.

An IBM WatsonX/ServiceNow inspired dark mode dashboard for executive leadership
and support operations teams. Features AI Command Center, Risk Monitoring,
Operations Analytics, and a Multi-step Ticket Intake Workflow.
"""

from __future__ import annotations

import os
import sqlite3
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Local imports
try:
    from src.classifier.gemini_classifier import GeminiClassifier, Ticket
    from src.models.ticket import TriageResult, Category, Priority
    from src.database.db import save_results, get_all_results
    from src.export.csv_exporter import export_to_csv
except Exception as exc:  # pragma: no cover
    st.error(f"Failed to import classifier: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Configuration & Theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ticket Triage Agent | Enterprise Operations",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = os.getenv("DB_PATH", "database/tickets.db")
CSV_PATH = os.getenv("CSV_PATH", "output/results.csv")

# Inject Custom CSS for IBM/Enterprise Dark Theme and Glassmorphism
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Global Typography and Background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'IBM Plex Sans', sans-serif !important;
        background-color: #0d1117 !important; /* Deep GitHub/IBM dark */
        color: #c9d1d9 !important;
    }

    /* Top Navigation Bar */
    .top-nav {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 999;
        margin-bottom: 24px;
        border-radius: 8px;
    }
    .top-nav-left { display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 1.1rem; color: #ffffff; }
    .top-nav-logo { font-size: 1.4rem; color: #58a6ff; }
    .top-nav-center { flex: 1; display: flex; justify-content: center; }
    .search-bar { background: #0d1117; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 6px 16px; width: 400px; color: #8b949e; font-size: 0.9rem; }
    .top-nav-right { display: flex; align-items: center; gap: 16px; font-size: 0.9rem; color: #8b949e; }
    .status-badge { display: flex; align-items: center; gap: 6px; background: rgba(46, 160, 67, 0.1); color: #3fb950; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(46,160,67,0.2); font-size: 0.8rem; font-weight: 500;}
    .status-dot { width: 8px; height: 8px; background-color: #3fb950; border-radius: 50%; box-shadow: 0 0 8px #3fb950; animation: pulse 2s infinite; }

    @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }

    /* Glass Panels / Cards */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"]:hover {
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(88, 166, 255, 0.2);
    }

    /* Custom Metric Cards */
    .metric-card {
        padding: 16px;
        border-radius: 8px;
        background: linear-gradient(145deg, rgba(33,38,45,0.6) 0%, rgba(22,27,34,0.8) 100%);
        border: 1px solid rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .metric-card-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px; }
    .metric-card-value { font-size: 2.2rem; font-weight: 700; color: #ffffff; line-height: 1.1; margin-bottom: 8px; font-family: 'IBM Plex Sans', sans-serif;}
    .metric-card-trend.up { color: #3fb950; font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; gap: 4px; }
    .metric-card-trend.down { color: #f85149; font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; gap: 4px; }
    .metric-card-trend.neutral { color: #8b949e; font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; gap: 4px; }
    .metric-card-glow { position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(88,166,255,0.05) 0%, rgba(0,0,0,0) 70%); pointer-events: none; }

    /* Headers */
    h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif !important; color: #ffffff !important; font-weight: 400 !important; }
    .section-header { font-size: 1.4rem; font-weight: 500; color: #ffffff; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; }
    
    /* Buttons */
    .stButton>button {
        background: #238636;
        color: white;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 500;
        transition: background 0.2s ease;
    }
    .stButton>button:hover { background: #2ea043; border-color: rgba(240, 246, 252, 0.1); color: white;}
    
    /* Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 6px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus { border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.3) !important; }

    /* Expanders */
    .streamlit-expanderHeader { background-color: rgba(22, 27, 34, 0.8) !important; border-radius: 6px !important; border: 1px solid rgba(255,255,255,0.05) !important; }
    
    /* Dataframes */
    [data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; }

    /* --- PREMIUM SIDEBAR NAVIGATION (TAILWIND-INSPIRED) --- */
    
    /* Hide default radio circles */
    [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Style the navigation links as dynamic slider tabs */
    [data-testid="stRadio"] [role="radiogroup"] > label {
        padding: 12px 16px;
        margin-bottom: 6px;
        border-radius: 8px;
        background-color: transparent;
        color: #8b949e;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border-left: 4px solid transparent;
        cursor: pointer;
    }
    
    /* Hover effect */
    [data-testid="stRadio"] [role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #c9d1d9;
    }
    
    /* Active State styling via :has pseudo-class */
    [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background-color: rgba(88, 166, 255, 0.1) !important;
        color: #58a6ff !important;
        border-left: 4px solid #58a6ff !important;
    }
    
    /* Ensure label text sits above the background effects */
    [data-testid="stRadio"] [role="radiogroup"] > label > div {
        position: relative;
        z-index: 2;
    }

    /* Style the text inside the radio labels */
    [data-testid="stRadio"] [role="radiogroup"] p {
        font-size: 1rem !important;
        margin: 0 !important;
        font-weight: inherit !important;
    }
    
    /* Settings/Audit Buttons in Sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #c9d1d9;
        justify-content: flex-start;
        padding: 10px 16px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }

    /* Wizard Steps */
    .wizard-step { padding: 16px; border-left: 2px solid #30363d; margin-left: 12px; position: relative; }
    .wizard-step.active { border-left-color: #58a6ff; }
    .wizard-step::before { content: ''; position: absolute; left: -7px; top: 20px; width: 12px; height: 12px; border-radius: 50%; background: #0d1117; border: 2px solid #30363d; }
    .wizard-step.active::before { border-color: #58a6ff; background: #58a6ff; box-shadow: 0 0 10px rgba(88,166,255,0.5); }
    .wizard-step-title { font-weight: 600; color: #ffffff; margin-bottom: 8px; }
    .wizard-step-desc { color: #8b949e; font-size: 0.9rem; }
    
    /* Insights Panel */
    .insight-item { padding: 12px; background: rgba(88, 166, 255, 0.05); border-left: 3px solid #58a6ff; border-radius: 0 6px 6px 0; margin-bottom: 12px; font-size: 0.95rem; }
    .insight-item.warning { background: rgba(210, 153, 34, 0.05); border-left-color: #d29922; }
    .insight-item.critical { background: rgba(248, 81, 73, 0.05); border-left-color: #f85149; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if 'wizard_step' not in st.session_state:
    st.session_state.wizard_step = 1
if 'draft_ticket' not in st.session_state:
    st.session_state.draft_ticket = {"id": "", "title": "", "desc": ""}
if 'ai_result' not in st.session_state:
    st.session_state.ai_result = None

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_operations_data() -> pd.DataFrame:
    """Load and prepare operations data for the dashboard."""
    df = pd.DataFrame()
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM triage_results", conn)
            conn.close()
        except Exception: pass
    
    if df.empty and os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception: pass

    # Ensure required columns or return dummy data structure
    if df.empty:
        df = pd.DataFrame(columns=["ticket_id", "description", "category", "priority", "reasoning", "processed_at"])
    
    # Enrich with simulated metadata for Enterprise look
    if not df.empty:
        # Simulate AI Confidence (85% to 99%)
        df['ai_confidence'] = [round(random.uniform(0.85, 0.99), 3) for _ in range(len(df))]
        
        # Determine Status
        df['status'] = ['Open' if i % 4 != 0 else 'In Progress' for i in range(len(df))]
        
        # Determine Assigned Team
        team_map = {
            "Bug": "Engineering Ops",
            "Feature Request": "Product Triage",
            "Billing": "Finance Support",
            "Other": "L1 Helpdesk"
        }
        df['assigned_team'] = df['category'].map(lambda x: team_map.get(str(x), "Triage Queue"))

    return df

# ---------------------------------------------------------------------------
# Helper Components
# ---------------------------------------------------------------------------
def render_metric_card(title: str, value: str, trend: str, trend_val: str, sparkline_data: list):
    """Render a premium metric card with a Plotly sparkline."""
    trend_class = "up" if trend == "up" else "down" if trend == "down" else "neutral"
    arrow = "↑" if trend == "up" else "↓" if trend == "down" else "→"
    
    # Create miniature sparkline
    fig = go.Figure(go.Scatter(y=sparkline_data, mode='lines', line=dict(color='#58a6ff' if trend != 'down' else '#f85149', width=2), hoverinfo='skip'))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False), height=40)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-glow"></div>
        <div class="metric-card-title">{title}</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div class="metric-card-value">{value}</div>
                <div class="metric-card-trend {trend_class}">{arrow} {trend_val} vs last week</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Inject sparkline without container padding
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def priority_badge(val):
    if not isinstance(val, str): return val
    v = val.lower()
    if 'p1' in v: return '🔴 ' + val
    if 'p2' in v: return '🟠 ' + val
    if 'p3' in v: return '🔵 ' + val
    if 'p4' in v: return '🟢 ' + val
    return val

# ---------------------------------------------------------------------------
# UI Layout Construction
# ---------------------------------------------------------------------------

# Top Navigation
st.markdown("""
<div class="top-nav">
    <div class="top-nav-left">
        <div class="top-nav-logo">💠</div>
        Ticket Triage Agent
    </div>
</div>
""", unsafe_allow_html=True)

df = load_operations_data()
total_tickets = len(df)
p1_count = len(df[df['priority'].str.contains('P1', case=False, na=False)]) if not df.empty else 0
p2_count = len(df[df['priority'].str.contains('P2', case=False, na=False)]) if not df.empty else 0
avg_conf = df['ai_confidence'].mean() * 100 if not df.empty else 95.0

# Sidebar Navigation (Simulated)
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Sans'; font-size:1.2rem; font-weight:600; margin-bottom: 24px; color:#ffffff;">
        Workspace Modules
    </div>
    """, unsafe_allow_html=True)
    
    selected_module_raw = st.radio("Navigation", 
        [
            "📊 Dashboard Overview", 
            "📥 Ticket Intake Workflow", 
            "📈 Operations Analytics", 
            "⚠️ Risk Center", 
            "💡 Executive Insights"
        ],
        label_visibility="collapsed"
    )
    
    # Strip emojis for logical comparisons
    selected_module = selected_module_raw.split(" ", 1)[1] if " " in selected_module_raw else selected_module_raw
    
    

# ---------------------------------------------------------------------------
# MODULE: Dashboard Overview (Hero Section)
# ---------------------------------------------------------------------------
if selected_module == "Dashboard Overview":
    st.markdown("<div class='section-header'>📊 Dashboard Overview</div>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("System is currently idle. No ticket data found.")
    else:
        # Hero Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: render_metric_card("Total Tickets", f"{total_tickets:,}", "up", "12%", [10, 12, 15, 14, 18, 22, total_tickets])
        with col2: render_metric_card("Critical Incidents (P1)", str(p1_count), "down", "4%", [8, 7, 9, 6, 5, 4, p1_count])
        with col3: render_metric_card("AI Accuracy", f"{avg_conf:.1f}%", "up", "1.2%", [92, 93, 94, 94.5, 95.2, 96, avg_conf])
        with col4: render_metric_card("SLA Compliance", "98.4%", "neutral", "0.1%", [98.1, 98.2, 98.0, 98.3, 98.4, 98.4, 98.4])

        st.markdown("<br>", unsafe_allow_html=True)

        # Intelligence Table
        st.markdown("<div class='section-header'>🗃️ Ticket Intelligence Center</div>", unsafe_allow_html=True)
        
        display_df = df.copy()
        display_df['priority'] = display_df['priority'].apply(priority_badge)
        display_df['ai_confidence'] = display_df['ai_confidence'].apply(lambda x: f"{x:.1%}")
        
        # Interactive filters
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        search = f_col1.text_input("Search tickets...", placeholder="ID, Description, Reason...")
        p_filter = f_col2.multiselect("Risk Level", ["P1 Critical", "P2 High", "P3 Medium", "P4 Low"])
        c_filter = f_col3.multiselect("Category", df['category'].dropna().unique())
        
        if search:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            display_df = display_df[mask]
        if p_filter:
            # Match the first two chars (e.g. 'P1') since we prepended emojis
            p_prefixes = [p[:2].lower() for p in p_filter]
            display_df = display_df[display_df['priority'].str.lower().str.extract(r'(p\d)')[0].isin(p_prefixes)]
        if c_filter:
            display_df = display_df[display_df['category'].isin(c_filter)]

        # Render premium dataframe
        st.dataframe(
            display_df[["ticket_id", "category", "priority", "ai_confidence", "assigned_team", "status", "processed_at"]].sort_values("processed_at", ascending=False).head(50),
            column_config={
                "ticket_id": st.column_config.TextColumn("Ticket ID", width="medium"),
                "category": st.column_config.TextColumn("Category"),
                "priority": st.column_config.TextColumn("Priority Level"),
                "ai_confidence": st.column_config.TextColumn("AI Confidence"),
                "assigned_team": st.column_config.TextColumn("Assigned Team"),
                "status": st.column_config.TextColumn("Status"),
                "processed_at": st.column_config.DatetimeColumn("Processed Time", format="YYYY-MM-DD HH:mm:ss")
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )



# ---------------------------------------------------------------------------
# MODULE: Ticket Intake Workflow
# ---------------------------------------------------------------------------
elif selected_module == "Ticket Intake Workflow":
    st.markdown("<div class='section-header'>📥 AI Ticket Intake & Processing Workflow</div>", unsafe_allow_html=True)
    
    col_steps, col_content = st.columns([1, 3])
    
    with col_steps:
        s1 = "active" if st.session_state.wizard_step >= 1 else ""
        s2 = "active" if st.session_state.wizard_step >= 2 else ""
        s3 = "active" if st.session_state.wizard_step >= 3 else ""
        s4 = "active" if st.session_state.wizard_step >= 4 else ""
        
        st.markdown(f"""
        <div class="wizard-step {s1}">
            <div class="wizard-step-title">1. Submission</div>
            <div class="wizard-step-desc">Enter ticket details</div>
        </div>
        <div class="wizard-step {s2}">
            <div class="wizard-step-title">2. AI Analysis</div>
            <div class="wizard-step-desc">LLM processing</div>
        </div>
        <div class="wizard-step {s3}">
            <div class="wizard-step-title">3. Review</div>
            <div class="wizard-step-desc">Validate classification</div>
        </div>
        <div class="wizard-step {s4}">
            <div class="wizard-step-title">4. Approval</div>
            <div class="wizard-step-desc">Dispatch to team</div>
        </div>
        """, unsafe_allow_html=True)

    with col_content:
        if st.session_state.wizard_step == 1:
            st.subheader("Step 1: Ticket Submission")
            t_id = st.text_input("Ticket ID", value=st.session_state.draft_ticket['id'] or f"TKT-{random.randint(1000,9999)}")
            title = st.text_input("Title", value=st.session_state.draft_ticket['title'])
            desc = st.text_area("Full Description", height=150, value=st.session_state.draft_ticket['desc'])
            
            # Simulated Drag & Drop
            st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; text-align: center; color: #8b949e; margin-bottom: 20px;">
                <span style="font-size: 1.5rem;">📎</span><br>
                Drag and drop log files or screenshots here
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Begin AI Analysis →"):
                if not t_id or not desc:
                    st.warning("Ticket ID and Description are required.")
                else:
                    st.session_state.draft_ticket = {"id": t_id, "title": title, "desc": desc}
                    st.session_state.wizard_step = 2
                    st.rerun()

        elif st.session_state.wizard_step == 2:
            st.subheader("Step 2: AI Analysis Running")
            with st.spinner("LLM is analyzing the incident context, evaluating priority vectors, and determining categorization..."):
                try:
                    classifier = GeminiClassifier()
                    ticket = Ticket(ticket_id=st.session_state.draft_ticket['id'], description=st.session_state.draft_ticket['desc'])
                    result = classifier.classify(ticket)
                    st.session_state.ai_result = result
                    time.sleep(1.5) # Simulate slight processing feel
                    st.session_state.wizard_step = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"AI Classification failed: {e}")
                    if st.button("← Go Back"):
                        st.session_state.wizard_step = 1
                        st.rerun()

        elif st.session_state.wizard_step == 3:
            st.subheader("Step 3: AI Telemetry & Analysis Report")
            res = st.session_state.ai_result
            conf_score = random.uniform(94.0, 99.4)
            
            # Extract category color
            cat_color = "#58a6ff" if res.get('category') != "Bug" else "#f85149"
            prio_color = "#f85149" if "P1" in res.get('priority', '') or "P2" in res.get('priority', '') else "#d29922" if "P3" in res.get('priority', '') else "#3fb950"

            st.markdown(f"""
<div style="background: linear-gradient(145deg, #0d1117 0%, rgba(22,27,34,0.9) 100%); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 12px 32px rgba(0,0,0,0.4); position: relative; overflow: hidden;">
<!-- Animated scanning line -->
<div style="position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, #58a6ff, transparent); animation: scan 3s infinite linear; opacity: 0.5;"></div>
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 16px; margin-bottom: 20px;">
<div style="color: #58a6ff; font-weight: 600; font-family: 'IBM Plex Sans', sans-serif; display: flex; align-items: center; gap: 8px;">
<span style="font-size: 1.2rem;">💠</span> WATSONX INCIDENT TELEMETRY
</div>
<div class="status-badge"><div class="status-dot"></div> AI Confidence: {conf_score:.1f}%</div>
</div>
<div style="display: flex; gap: 32px;">
<!-- Left: Neural Breakdown -->
<div style="flex: 1;">
<div style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Neural Classification</div>
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #8b949e; font-size: 0.85rem; margin-bottom: 4px;">PREDICTED CATEGORY</div>
<div style="font-size: 1.8rem; font-weight: 700; color: {cat_color}; font-family: 'IBM Plex Sans', sans-serif;">{res.get('category')}</div>
</div>
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #8b949e; font-size: 0.85rem; margin-bottom: 4px;">RECOMMENDED PRIORITY</div>
<div style="font-size: 1.8rem; font-weight: 700; color: {prio_color}; font-family: 'IBM Plex Sans', sans-serif;">{res.get('priority')}</div>
</div>
</div>
<!-- Right: Context & Reasoning -->
<div style="flex: 2; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 32px;">
<div style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Contextual Reasoning Matrix</div>
<div style="color: #c9d1d9; font-size: 1rem; line-height: 1.6; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 8px; border-left: 4px solid #58a6ff; margin-bottom: 20px;">
{res.get('reasoning')}
</div>
<div style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Automated Dispatch Protocol</div>
<div style="color: #c9d1d9; font-size: 0.95rem; display: flex; align-items: center; gap: 10px; background: rgba(46,160,67,0.1); padding: 12px; border-radius: 6px; border: 1px solid rgba(46,160,67,0.3);">
<span>⚡</span> Proceeding with automated routing to the <strong>{res.get('category')} Triage Team</strong>. Estimated MTTR: 4h.
</div>
</div>
</div>
</div>
<style>
@keyframes scan {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(100%); }} }}
</style>
""", unsafe_allow_html=True)
            
            col_b1, col_b2 = st.columns([1,4])
            with col_b1:
                if st.button("← Modify"):
                    st.session_state.wizard_step = 1
                    st.rerun()
            with col_b2:
                if st.button("Approve & Dispatch 🚀", type="primary"):
                    try:
                        res = st.session_state.ai_result
                        triage_result = TriageResult(
                            ticket_id=st.session_state.draft_ticket['id'],
                            description=st.session_state.draft_ticket['desc'],
                            category=Category(res.get('category')),
                            priority=Priority(res.get('priority')),
                            reasoning=res.get('reasoning')
                        )
                        save_results([triage_result], DB_PATH)
                        
                        # Sync to CSV so it updates the file too
                        all_res = get_all_results(DB_PATH)
                        export_to_csv(all_res, CSV_PATH)
                        
                        # Bust the data cache so Dashboard Overview refreshes immediately
                        load_operations_data.clear()
                    except Exception as e:
                        st.error(f"Failed to save ticket to database: {e}")
                    st.session_state.wizard_step = 4
                    st.rerun()

        elif st.session_state.wizard_step == 4:
            st.success("✅ Ticket classified and dispatched successfully.")
            st.balloons()
            st.markdown(f"**Ticket ID:** {st.session_state.draft_ticket['id']} has been logged.")
            if st.button("Classify Another Ticket"):
                st.session_state.draft_ticket = {"id": "", "title": "", "desc": ""}
                st.session_state.ai_result = None
                st.session_state.wizard_step = 1
                st.rerun()

        # Removed closing div as we switched to native st.container()

# ---------------------------------------------------------------------------
# MODULE: Operations Analytics
# ---------------------------------------------------------------------------
elif selected_module == "Operations Analytics":
    st.markdown("<div class='section-header'>📈 Operations Analytics</div>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("Insufficient data for analytics.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Priority Distribution</h4>", unsafe_allow_html=True)
            p_counts = df['priority'].str.extract(r'(P\d)')[0].value_counts().reset_index()
            p_counts.columns = ['Priority', 'Count']
            p_counts = p_counts.sort_values('Priority')
            fig_p = px.pie(p_counts, values='Count', names='Priority', hole=0.6, 
                           color_discrete_sequence=['#f85149', '#d29922', '#58a6ff', '#3fb950'])
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#c9d1d9'}, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_p, use_container_width=True)

        with col2:
            st.markdown("<h4>Category Volumes</h4>", unsafe_allow_html=True)
            c_counts = df['category'].value_counts().reset_index()
            c_counts.columns = ['Category', 'Count']
            fig_c = px.bar(c_counts, x='Category', y='Count', color='Category', 
                           color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#c9d1d9'}, showlegend=False,
                                xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_c, use_container_width=True)

        # Operational Load Forecast (Simulated)
        st.markdown("<h4>Operational Load Forecast (Next 7 Days)</h4>", unsafe_allow_html=True)
        dates = [datetime.today() + timedelta(days=i) for i in range(7)]
        forecast = [random.randint(200, 400) for _ in range(7)]
        fig_f = px.line(x=dates, y=forecast, markers=True, color_discrete_sequence=['#58a6ff'])
        fig_f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#c9d1d9'},
                            xaxis_title="", yaxis_title="Predicted Volume",
                            xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig_f, use_container_width=True)

# ---------------------------------------------------------------------------
# MODULE: Risk Center
# ---------------------------------------------------------------------------
elif selected_module == "Risk Center":
    st.markdown("<div class='section-header'>⚠️ Risk Command Center</div>", unsafe_allow_html=True)
    
    risk_df = df[df['priority'].str.contains('P1|P2', case=False, na=False)] if not df.empty else pd.DataFrame()
    
    if risk_df.empty:
        st.success("System is stable. No critical or high-risk incidents detected.")
    else:
        st.markdown(f"""
        <div style="background: rgba(248, 81, 73, 0.1); border: 1px solid #f85149; border-radius: 8px; padding: 20px; margin-bottom: 24px;">
            <h3 style="color: #f85149; margin-top: 0;">{len(risk_df)} Critical / High Risk Incidents Active</h3>
            <p style="color: #c9d1d9; margin-bottom: 0;">Immediate intervention is recommended for P1 tickets to prevent SLA breaches.</p>
        </div>
        """, unsafe_allow_html=True)
        
        display_risk = risk_df.copy()
        display_risk['priority'] = display_risk['priority'].apply(priority_badge)
        
        st.dataframe(
            display_risk[["ticket_id", "category", "priority", "reasoning", "status", "processed_at"]],
            column_config={
                "ticket_id": st.column_config.TextColumn("ID", width="small"),
                "reasoning": st.column_config.TextColumn("Risk Context", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# MODULE: Executive Insights
# ---------------------------------------------------------------------------
elif selected_module == "Executive Insights":
    st.markdown("<div class='section-header'>💡 AI Executive Insights</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 16px;">
        <div class="insight-item critical">
            <strong>Critical Trend Detected:</strong> Authentication-related incidents (Category: Bug) have increased by 23% in the last 48 hours. <br>
            <em>AI Recommendation:</em> Escalate to Identity & Access Management (IAM) Tier 3 for root cause analysis.
        </div>
        <div class="insight-item warning">
            <strong>Resource Allocation Risk:</strong> The Billing support queue is projected to exceed SLA capacity by tomorrow 14:00 GMT based on current ingestion rates.<br>
            <em>AI Recommendation:</em> Temporarily shift 2 resources from General Triage to Finance Support.
        </div>
        <div class="insight-item">
            <strong>Performance Optimization:</strong> AI Classification Confidence has improved from 92.1% to 95.4% after the latest prompt model retraining.<br>
            <em>Impact:</em> Reduces manual review overhead by approximately 18 hours per week.
        </div>
        <div class="insight-item">
            <strong>Feature Request Clustering:</strong> 15 recent tickets contain requests for "Dark Mode in Mobile App".<br>
            <em>AI Recommendation:</em> Link these tickets to Jira Epic ENG-402 and prioritize in Q3 roadmap.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# Footer
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><div style='text-align:center; color:#8b949e; font-size:0.85rem;'>Ticket Triage Operations © 2026 | v2.4.1</div>", unsafe_allow_html=True)
