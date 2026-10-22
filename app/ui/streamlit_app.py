from pathlib import Path
import sys

# Streamlit executes this file directly; make package imports independent of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import os
import pandas as pd
from app.ui.backend import call_api

# Optional separate backend; unset API_URL runs the analytics in this process.
API_URL = os.getenv("API_URL", "")

# Page configuration
st.set_page_config(
    page_title="IPL Analytics Assistant",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #fdfdfd;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .header-box::after {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background: radial-gradient(circle at 80% 20%, rgba(255, 107, 0, 0.15) 0%, transparent 50%);
        pointer-events: none;
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    .status-online {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-offline {
        background-color: #ffedd5;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }
    
    /* Info box */
    .info-card {
        background-color: #f8fafc;
        border-left: 5px solid #ff6b00;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .info-card-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .info-card-content {
        color: #475569;
        font-size: 0.95rem;
    }
    
    /* Stat Box card */
    .stat-box {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .stat-val {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ff6b00;
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Cache health status check
@st.cache_data(ttl=10)
def check_api_health():
    try:
        return call_api("/health")
    except Exception:
        pass
    return None

# Render Sidebar
def render_sidebar(health_data):
    with st.sidebar:
        st.markdown("# 🏏")
        st.markdown("### IPL Analytics Hub")
        st.markdown("Explore player stats, historical venue profiles, match phases, and head-to-head match-up details.")
        
        # System Health Status Widget
        st.markdown("---")
        st.markdown("#### System Health")
        if health_data:
            if health_data.get("answer_mode") == "analytics":
                st.markdown("⚡ Analytics answers · no model required")
            elif health_data.get("model_loaded"):
                st.markdown('<div class="status-badge status-online">🟢 LLM Generator: Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-offline">⚡ Retrieval-Only Mode (CPU)</div>', unsafe_allow_html=True)
            
            st.markdown(f"📊 **Data Source:** Available ({health_data.get('analytics_tables_count', 0)} tables)")
            st.markdown(f"🧠 **Insights Corpus:** {health_data.get('insights_count', 0)} entries")
        else:
            st.markdown('<div class="status-badge status-offline">🔴 API Server: Offline</div>', unsafe_allow_html=True)
            st.warning("Cannot connect to FastAPI backend at " + API_URL)
            
        # Sample Queries Selection
        st.markdown("---")
        st.markdown("#### Quick Suggestions")
        suggestions = [
            "Who are the top IPL run scorers?",
            "What are V Kohli's IPL career stats?",
            "Compare Bumrah and Chahal.",
            "V Kohli vs JJ Bumrah head-to-head",
            "What are the stats for Chinnaswamy Stadium?",
            "Which phase has the highest run rate?",
            "Why is boundary percentage important?",
            "Is SP Narine economical?"
        ]
        
        st.markdown("Click any query to copy it:")
        for sug in suggestions:
            if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                st.session_state.query_input = sug
                st.rerun()

# ── Main Header ──
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🏏 IPL Analytics Assistant</h1>
    <p class="header-subtitle">Domain-Specific RAG System powered by clean Parquet data, pre-computed cricket analytics, and searchable historical insights.</p>
</div>
""", unsafe_allow_html=True)

# Fetch health information
health_info = check_api_health()
render_sidebar(health_info)

# Initialize Session State for query input
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Input row
query = st.text_input("Ask a question:", value=st.session_state.query_input, placeholder="e.g. V Kohli vs JJ Bumrah head-to-head")

col_ask, col_clear, _ = st.columns([1, 1, 8])
ask_clicked = col_ask.button("Ask", type="primary", use_container_width=True)
clear_clicked = col_clear.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.query_input = ""
    st.rerun()

# ── Query execution ──
if ask_clicked and query.strip():
    with st.spinner("Processing query..."):
        # 1. Fetch retrieval debug traces in parallel first (always safe/fast)
        retrieve_data = None
        try:
            retrieve_data = call_api("/debug/retrieve", {"query": query})
        except Exception as e:
            st.error(f"Failed to query retrieval layer: {e}")
            
        # 2. Try fetching full LLM chat generation (if available)
        chat_data = None
        use_llm_fallback = True
        
        # Only call /chat if the model is active or we want to try it
        if health_info and (health_info.get("answer_mode") == "analytics" or health_info.get("model_loaded")):
            try:
                chat_data = call_api("/chat", {"query": query})
                use_llm_fallback = False
            except Exception:
                st.warning("The answer could not be prepared. Retrieved facts are shown below.")
        
        if retrieve_data:
            intent = retrieve_data.get("intent", "general")
            route = retrieve_data.get("route", "structured")
            entities = retrieve_data.get("entities", {})
            
            # --- Rendering Answer Header ---
            st.markdown("### 📝 Analysis & Insights")
            
            if not use_llm_fallback and chat_data:
                # Beautiful card for LLM answer
                st.markdown(chat_data.get("answer"))
            else:
                # LLM offline: Render our beautiful custom Structured / Context visualization
                st.markdown(
                    '<div class="info-card">'
                    '<div class="info-card-title">⚡ CPU Retrieval Fallback</div>'
                    '<div class="info-card-content">The LLM generation model is offline. '
                    'Below is the live, pre-computed exact statistical output and computed matchup metrics from the data pipeline.</div>'
                    '</div>', 
                    unsafe_allow_html=True
                )
                
                # Render the raw pre-computed context as a clean text card
                st.markdown("#### Pre-computed Analytical Summary")
                st.code(retrieve_data.get("context", "No analytical context generated."), language="markdown")
            
            # --- Rendering Entity Tags ---
            st.markdown("---")
            st.markdown("#### Identified Entities")
            cols = st.columns(5)
            with cols[0]:
                st.markdown(f"**🏏 Batters:** {', '.join(entities.get('batters', [])) or 'None'}")
            with cols[1]:
                st.markdown(f"**🥎 Bowlers:** {', '.join(entities.get('bowlers', [])) or 'None'}")
            with cols[2]:
                st.markdown(f"**📅 Seasons:** {', '.join(map(str, entities.get('seasons', []))) or 'None'}")
            with cols[3]:
                st.markdown(f"**🏟 Venues:** {', '.join(entities.get('venues', [])) or 'None'}")
            with cols[4]:
                st.markdown(f"**⏱ Phases:** {', '.join(entities.get('phases', [])) or 'None'}")

            # --- Rendering Interactive Charts & Metrics based on data ---
            structured_results = retrieve_data.get("structured_results", [])
            insight_results = retrieve_data.get("insight_results", [])
            
            if structured_results:
                st.markdown("---")
                st.markdown("### 📊 Live Analytics & Visualizations")
                
                # Check for Matchup Data
                matchups = [r for r in structured_results if r.get("type") == "matchup"]
                batter_stats = [r for r in structured_results if r.get("type") in ("batter_career", "batter_season")]
                bowler_stats = [r for r in structured_results if r.get("type") in ("bowler_career", "bowler_season")]
                
                if matchups:
                    m = matchups[0]["data"]
                    st.subheader(f"⚔ Head-to-Head: {m['batter']} vs {m['bowler']}")
                    
                    # Highlight Key derived metrics
                    m_cols = st.columns(4)
                    with m_cols[0]:
                        st.markdown(
                            f'<div class="stat-box"><div class="stat-val">{m["runs"]}</div><div class="stat-lbl">Runs scored</div></div>', 
                            unsafe_allow_html=True
                        )
                    with m_cols[1]:
                        st.markdown(
                            f'<div class="stat-box"><div class="stat-val">{m["balls"]}</div><div class="stat-lbl">Balls faced</div></div>', 
                            unsafe_allow_html=True
                        )
                    with m_cols[2]:
                        st.markdown(
                            f'<div class="stat-box"><div class="stat-val">{m["strike_rate"]:.1f}</div><div class="stat-lbl">Strike Rate</div></div>', 
                            unsafe_allow_html=True
                        )
                    with m_cols[3]:
                        st.markdown(
                            f'<div class="stat-box"><div class="stat-val">{m["wickets"]}</div><div class="stat-lbl">Dismissals</div></div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Chart representation
                    st.markdown("<br>", unsafe_allow_html=True)
                    chart_data = pd.DataFrame({
                        "Metric": ["Dots", "Fours", "Sixes"],
                        "Count": [m["dots"], m["fours"], m["sixes"]]
                    })
                    st.bar_chart(chart_data.set_index("Metric"))

                elif len(batter_stats) >= 2:
                    # Side-by-side comparison of batters
                    st.subheader("🏏 Batter Comparison Profile")
                    
                    compare_df = pd.DataFrame([b["data"] for b in batter_stats]).set_index("batter")
                    # Display clean comparison table
                    columns_to_show = ["seasons", "matches", "runs", "strike_rate", "batting_average", "boundary_pct", "dot_pct"]
                    st.dataframe(compare_df[columns_to_show], use_container_width=True)
                    
                    # Side-by-side bar chart
                    chart_cols = st.columns(2)
                    with chart_cols[0]:
                        st.markdown("**Runs Scored**")
                        st.bar_chart(compare_df["runs"])
                    with chart_cols[1]:
                        st.markdown("**Strike Rate comparison**")
                        st.bar_chart(compare_df["strike_rate"])
                        
                elif len(bowler_stats) >= 2:
                    # Side-by-side comparison of bowlers
                    st.subheader("🥎 Bowler Comparison Profile")
                    compare_df = pd.DataFrame([b["data"] for b in bowler_stats]).set_index("bowler")
                    columns_to_show = ["seasons", "matches", "wickets", "economy", "bowling_strike_rate", "bowling_average", "dot_pct"]
                    st.dataframe(compare_df[columns_to_show], use_container_width=True)
                    
                    chart_cols = st.columns(2)
                    with chart_cols[0]:
                        st.markdown("**Wickets Taken**")
                        st.bar_chart(compare_df["wickets"])
                    with chart_cols[1]:
                        st.markdown("**Economy Rate (lower is better)**")
                        st.bar_chart(compare_df["economy"])

                else:
                    # Generic display for single records or other data tables
                    for res in structured_results:
                        res_type = res.get("type", "data")
                        res_data = res.get("data", {})
                        
                        if res_type == "batter_career":
                            st.subheader(f"🏏 Batting Profile: {res_data.get('batter')}")
                            df = pd.DataFrame([res_data]).set_index("batter")
                            st.dataframe(df, use_container_width=True)
                        elif res_type == "bowler_career":
                            st.subheader(f"🥎 Bowling Profile: {res_data.get('bowler')}")
                            df = pd.DataFrame([res_data]).set_index("bowler")
                            st.dataframe(df, use_container_width=True)
                        elif res_type.endswith("leaderboard"):
                            metric = res.get("metric", "metric")
                            rank = res.get("rank", 1)
                            st.markdown(f"**Rank #{rank}** — Metric: {metric}")
                            st.json(res_data)
                        else:
                            st.markdown(f"#### Data Profile: {res_type.replace('_', ' ').title()}")
                            st.json(res_data)
            
            # --- Rendering Semantic Insights ---
            if insight_results:
                st.markdown("---")
                st.markdown("### 💡 Semantic Search Insights")
                for ins in insight_results:
                    score = ins.get('score', 0.0)
                    st.markdown(
                        f'<div style="background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 1rem; border-radius: 6px; margin-bottom: 0.75rem;">'
                        f'<strong>Category: {ins.get("category", "General").title()}</strong> (Similarity Score: {score:.2f})<br>'
                        f'<span style="color: #334155; font-style: italic;">"{ins.get("insight")}"</span>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
            # Details Expander at the bottom
            with st.expander("🔍 RAG System Debug details"):
                st.write(f"**Internal Intent:** {intent} | **Selected Route:** {route}")
                st.write(f"**Structured Records Count:** {len(structured_results)} | **Insight Corpus Hits:** {len(insight_results)}")
                st.markdown("**Generated Context Template:**")
                st.code(retrieve_data.get("context", ""), language="markdown")
        else:
            st.error("No data could be retrieved from the backend API. Please make sure the FastAPI server is running on " + API_URL)
