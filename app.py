"""Streamlit front-end for the Synthetic Control with Differencing reproduction."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PANEL_PATH = DATA_DIR / "italy_nuts3_panel.parquet"
EVENT_STUDY_PATH = DATA_DIR / "event_study.png"
WEIGHTS_PATH = DATA_DIR / "weights.png"

TITLE = "Synthetic Control with Differencing"
SUBTITLE = "Italian Earthquake Application"

ARXIV_URL = "https://arxiv.org/abs/2510.26106"
REPOSITORY_URL = "https://github.com/Nikkaran/scd-italy-reproduction"

ATT_EMPLOYMENT = 0.2096
PLACEBO_P_VALUE = 0.0769
SIGNIFICANCE_LEVEL = 0.10

# Colors
GOLD = "#C5A059"
GOLD_LIGHT = "#E3C16F"
BG_DARK = "rgba(0,0,0,0)" # Transparent for plotly

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup: return lookup[candidate]
    for candidate in candidates:
        for lowered, original in lookup.items():
            if candidate in lowered: return original
    return None

@st.cache_data(show_spinner=False)
def load_panel(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def format_year(value) -> str:
    try: return str(int(value))
    except (TypeError, ValueError): return str(value)

def local_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0E1117;
    --panel: rgba(255, 255, 255, 0.03);
    --border: rgba(255, 255, 255, 0.08);
    --gold: #C5A059;
    --gold-light: #E3C16F;
    --text: #E8EAED;
    --muted: #9BA3AF;
    --ease: cubic-bezier(0.22, 1, 0.36, 1);
}

.stApp {
    background-image: 
        radial-gradient(1000px 600px at 10% 0%, rgba(197, 160, 89, 0.08), transparent 60%),
        radial-gradient(800px 500px at 90% 100%, rgba(84, 132, 204, 0.05), transparent 50%);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.block-container { padding-top: 2.5rem; max-width: 1100px; }

h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; color: #F3F4F6; }
h1 { font-weight: 700; font-size: 2.8rem; margin-bottom: 0.2rem; }
a { color: var(--gold-light); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Chrome off */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(11, 14, 19, 0.95);
    border-right: 1px solid var(--border);
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 240ms var(--ease), border-color 240ms var(--ease);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(197, 160, 89, 0.4);
}
div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetricValue"] { color: var(--gold-light); font-weight: 600; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.05);
}
.badge-gold { border-color: rgba(197, 160, 89, 0.4); color: var(--gold-light); background: rgba(197, 160, 89, 0.1); }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# UI Sections
# --------------------------------------------------------------------------- #
def render_header():
    st.title(TITLE)
    st.subheader(SUBTITLE)
    
    # Premium Badges
    st.markdown(f"""
        <div>
            <span class="badge badge-gold">Independent Reproduction</span>
            <span class="badge">arXiv:2510.26106</span>
            <span class="badge">p-value: {PLACEBO_P_VALUE}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
This dashboard presents an independent reproduction of **Synthetic Control with Differencing (SCD)**. 
The application evaluates the economic impact of the **May 2012 Emilia earthquake** on the province of **Modena**, using a synthetic control composed of 25 other Italian provinces.
""")

def render_results():
    st.divider()
    left, right = st.columns(2)
    left.metric("ATT on Employment", f"{ATT_EMPLOYMENT:.4f}", help="Average Treatment Effect on the Treated (log employment).")
    right.metric("Placebo p-value", f"{PLACEBO_P_VALUE:.4f}", help="In-space placebo inference.")
    
    st.caption("The estimate is statistically significant at the 10% level. The positive effect is driven by the massive post-earthquake reconstruction transfers stimulating local demand.")

def render_figures():
    st.divider()
    st.header("Visual Analysis")
    
    tabs = st.tabs(["📈 Effect Path", "⚖️ Donor Weights"])
    
    with tabs[0]:
        if EVENT_STUDY_PATH.exists():
            st.image(str(EVENT_STUDY_PATH), use_column_width=True)
            st.caption("Estimated effect on Modena's employment. The grey lines represent the placebo runs (where other provinces are artificially treated).")
        else:
            st.warning("Figure missing. Run `make_plots.py`.")

    with tabs[1]:
        if WEIGHTS_PATH.exists():
            st.image(str(WEIGHTS_PATH), use_column_width=True)
            st.caption("Simplex weights assigned to the donor provinces.")
        else:
            st.warning("Figure missing. Run `make_plots.py`.")

def create_plotly_chart(df, x_col, y_col, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], 
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=4, color=color),
        name=title
    ))
    fig.update_layout(
        title=title,
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font=dict(color='#E8EAED', family="Inter"),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=""),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    return fig

def render_explorer():
    st.divider()
    st.header("Interactive Data Explorer")

    if not PANEL_PATH.exists():
        st.warning("Data file not found.")
        return

    df = load_panel(str(PANEL_PATH))
    
    # Find columns
    geo_col = find_column(df.columns, ("geo", "province"))
    year_col = find_column(df.columns, ("year",))
    emp_col = find_column(df.columns, ("employment",))
    gdp_col = find_column(df.columns, ("gdp_pc", "gdp"))

    provinces = sorted(df[geo_col].dropna().astype(str).unique())
    selected = st.selectbox("Select a province to explore its raw data", provinces, index=provinces.index("ITH54") if "ITH54" in provinces else 0)

    subset = df[df[geo_col].astype(str) == selected].sort_values(year_col)

    # Plotly Tabs
    tab1, tab2, tab3 = st.tabs(["Employment Trend", "GDP Trend", "Raw Data"])
    
    with tab1:
        st.plotly_chart(create_plotly_chart(subset, year_col, emp_col, "Employment", GOLD), use_container_width=True)
    with tab2:
        st.plotly_chart(create_plotly_chart(subset, year_col, gdp_col, "GDP per Capita", "#5484CC"), use_container_width=True)
    with tab3:
        st.dataframe(subset, use_container_width=True, hide_index=True)

def render_sidebar():
    st.sidebar.title("SCD Italy")
    st.sidebar.caption("Research Portfolio")
    st.sidebar.divider()
    
    st.sidebar.markdown(f"[📄 Read the Paper]({ARXIV_URL})")
    st.sidebar.markdown(f"[💻 View Source on GitHub]({REPOSITORY_URL})")
    
    st.sidebar.divider()
    st.sidebar.markdown("**Methodology**")
    st.sidebar.info("""
    - **Treated:** Modena
    - **Shock:** 2012 Earthquake
    - **Donors:** 25 Provinces
    - **Method:** SLSQP Simplex
    """)

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="SCD Italy", layout="wide", initial_sidebar_state="expanded")
    local_css()
    render_sidebar()
    render_header()
    render_results()
    render_figures()
    render_explorer()

if __name__ == "__main__":
    main()