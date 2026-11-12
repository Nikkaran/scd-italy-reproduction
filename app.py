import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="SCD Italy | Research", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    
    /* Background & Layout */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(197, 160, 89, 0.05), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(84, 132, 204, 0.05), transparent 25%);
        background-color: #0E1117;
    }
    .block-container { padding-top: 3rem; max-width: 1100px; }
    
    /* Typography */
    h1 { font-weight: 700; color: #F3F4F6; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
    h2 { font-weight: 600; color: #E8EAED; margin-top: 2rem;}
    p, li { color: #9BA3AF; line-height: 1.6; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(197, 160, 89, 0.5);
    }
    div[data-testid="stMetricLabel"] { color: #C5A059 !important; font-weight: 500;}
    div[data-testid="stMetricValue"] { color: #F3F4F6 !important; font-weight: 700; font-size: 2.5rem;}
    
    /* Custom Badge */
    .custom-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
        background: rgba(197, 160, 89, 0.1); color: #C5A059; border: 1px solid rgba(197, 160, 89, 0.3);
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px; margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    path = Path("data/italy_nuts3_panel.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None

df = load_data()

# --- Sidebar ---
with st.sidebar:
    st.title("SCD Italy")
    st.markdown("Independent reproduction of **arXiv:2510.26106** applied to the 2012 Emilia Earthquake.")
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("[📄 Original Paper](https://arxiv.org/abs/2510.26106)")
    st.markdown("[💻 GitHub Repository](https://github.com/Nikkaran/scd-italy-reproduction)")
    st.markdown("---")
    st.info("**Methodology:** Synthetic Control with Differencing (SCD)\n\n**Treated Unit:** Modena (ITH54)\n\n**Inference:** In-space placebo")

# --- Main Content ---
st.markdown('<span class="custom-badge">Independent Reproduction</span> <span class="custom-badge">Causal Inference</span>', unsafe_allow_html=True)
st.title("Synthetic Control with Differencing")
st.markdown("### Italian Earthquake Application")
st.write("This dashboard presents a from-scratch implementation of the SCD estimator, evaluating the long-term economic impact of the May 2012 Emilia earthquake on Modena's employment rate.")

# --- Results Section ---
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("ATT on Employment (Log)", "0.2096", help="Average Treatment Effect on the Treated")
with col2:
    st.metric("Placebo P-Value", "0.0769", help="Statistically significant at the 10% level")

st.info("**Interpretation:** The massive post-earthquake reconstruction effort (government and EU transfers) acted as a positive demand shock, resulting in a statistically significant increase in long-term local employment, more than offsetting the initial destruction.")

# --- Figures ---
st.markdown("---")
st.header("Visual Analysis")
tab1, tab2 = st.tabs(["📈 Event Study", "⚖️ Donor Weights"])
with tab1:
    if Path("data/event_study.png").exists():
        st.image("data/event_study.png", use_column_width=True)
    else:
        st.warning("Event study plot not found.")
with tab2:
    if Path("data/weights.png").exists():
        st.image("data/weights.png", use_column_width=True)
    else:
        st.warning("Weights plot not found.")

# --- Data Explorer ---
st.markdown("---")
st.header("Interactive Data Explorer")

if df is not None and not df.empty:
    provinces = df['geo'].unique()
    selected_prov = st.selectbox("Select a province to view its historical data:", provinces, index=list(provinces).index("ITH54") if "ITH54" in provinces else 0)
    
    prov_data = df[df['geo'] == selected_prov].sort_values('year')
    
    fig = px.line(prov_data, x='year', y='employment', title=f"Employment Trend: {selected_prov}")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9BA3AF", family="Inter"),
        xaxis=dict(showgrid=False, title="Year"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Employment (Thousands)"),
        hovermode="x unified"
    )
    fig.update_traces(line_color="#C5A059", line_width=3)
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Data file not found. Please ensure `data/italy_nuts3_panel.parquet` is uploaded.")