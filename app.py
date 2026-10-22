import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()
st.set_page_config(page_title="NEXUS AI Data Analyst", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root { --neon-purple:#a855f7; --neon-cyan:#22d3ee; --neon-pink:#f472b6; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.stApp {
    background: radial-gradient(ellipse at top left,#321052,transparent 45%),
                radial-gradient(ellipse at top right,#102B49,transparent 40%),#090613;
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#190B2D,#10162B);
    border-right: 1px solid #A855F7;
}
.hero {
    padding: 28px; border-radius: 20px; border: 1px solid #A855F7;
    background: linear-gradient(110deg,#291044,#101E3D);
    box-shadow: 0 0 22px #7C3AED55, inset 0 0 24px #22D3EE0D;
    margin-bottom: 24px; position: relative; overflow: hidden;
    animation: heroGlow 7s ease-in-out infinite alternate;
}
.hero:after { content:""; position:absolute; inset:0; pointer-events:none;
    background:linear-gradient(115deg,transparent 20%,#ffffff0a 48%,transparent 70%); }
@keyframes heroGlow { from { box-shadow:0 0 16px #7C3AED44; } to { box-shadow:0 0 30px #22D3EE33,0 0 42px #A855F722; } }
.stApp [data-testid="stVerticalBlockBorderWrapper"] { border-color:#8b5cf655; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #a855f744; }
.stTabs [data-baseweb="tab"] { background:#160d27; border:1px solid #8b5cf644; border-radius:10px 10px 0 0; padding:10px 16px; }
.stTabs [aria-selected="true"] { background:linear-gradient(120deg,#6d28d9,#312e81) !important; border-color:#22d3ee !important; }
[data-testid="stFileUploader"] { background:#130d22; border:1px dashed #a855f7; border-radius:14px; padding:12px; }
[data-testid="stDataFrame"] { border:1px solid #8b5cf655; border-radius:12px; overflow:hidden; }
.stSelectbox [data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea { background:#100a1d; color:#f5f3ff; border-color:#8b5cf6; border-radius:10px; }
.stButton > button, [data-testid="stDownloadButton"] button { transition:all .2s ease; }
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover { transform:translateY(-1px); box-shadow:0 0 20px #a855f766; }
[data-testid="stAlert"] { border-radius:12px; }
.neon-strip { height:3px; margin: -8px 0 22px; border-radius:8px; background:linear-gradient(90deg,#a855f7,#22d3ee,#f472b6,#a855f7); background-size:200% 100%; animation: stripFlow 5s linear infinite; }
@keyframes stripFlow { to { background-position:200% 0; } }
.hero h1 {
    background: linear-gradient(90deg,#D8B4FE,#22D3EE,#F472B6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 38px; font-weight: 800;
}
div[data-testid="stMetric"] {
    background: #19102B; border: 1px solid #7945A5;
    border-radius: 14px; padding: 15px;
}
div[data-testid="stMetricValue"] { color: #67E8F9; }
.stButton > button {
    background: linear-gradient(90deg,#7C3AED,#A855F7,#6D28D9);
    color: white; border: 1px solid #C084FC; border-radius: 10px;
    font-weight: bold; width: 100%;
}
.stButton > button:hover {
    border-color: #22D3EE; box-shadow: 0 0 18px #22D3EE66;
}
h2,h3 { color: #D8B4FE; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>⚡ NEXUS AI — DATA ANALYST TEAM</h1>
<p>Advanced Multi-Agent Analytics | CrewAI | Google Gemini</p>
<p>Transform your CSV data into actionable business insights.</p>
</div>
<div class="neon-strip"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 NEXUS CONTROL")
    st.markdown("---")
    st.markdown("🟣 **Agent 1:** Data Analyst")
    st.markdown("🔵 **Agent 2:** Insight Analyst")
    st.markdown("🩷 **Agent 3:** Report Writer")
    st.markdown("---")
    model_name = st.text_input(
        "Gemini model",
        value=os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
    ).strip()
    st.caption("Powered by CrewAI + Google Gemini")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start your AI analysis.")
    c1, c2, c3 = st.columns(3)
    c1.info("🟣 Data Analyst\n\nProfiles your dataset.")
    c2.info("🔵 Insight Analyst\n\nFinds patterns and opportunities.")
    c3.info("🩷 Report Writer\n\nCreates a business report.")
    st.stop()

try:
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("Maximum supported file size is 20 MB.")
        st.stop()
    df = pd.read_csv(uploaded_file, low_memory=False)
    if df.empty:
        st.error("The CSV contains no data rows.")
        st.stop()
    if len(df) > 200000 or len(df.columns) > 100:
        st.error("Maximum supported size: 200,000 rows and 100 columns.")
        st.stop()
    df.columns = [
        str(c).strip() if str(c).strip() else f"column_{i}"
        for i, c in enumerate(df.columns)
    ]
    if df.columns.duplicated().any():
        st.error("Duplicate column names found. Please rename them.")
        st.stop()
except Exception as exc:
    st.error(f"CSV reading error: {exc}")
    st.stop()

st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{len(df):,}")
c2.metric("Total Columns", f"{len(df.columns):,}")
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗃️ Data Explorer", "🧪 Data Quality",
    "📈 Visual Analytics", "🤖 AI Analyst Team"
])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "⬇️ Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
        "dataset_export.csv", "text/csv"
    )
    schema = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Unique": df.nunique().values
    })
    st.subheader("Column Information")
    st.dataframe(schema, use_container_width=True)

with tab2:
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing = missing[missing["Missing Values"] > 0]
    if not missing.empty:
        fig = px.bar(
            missing, x="Column", y="Missing Values",
            color="Missing Values",
            color_continuous_scale=["#22D3EE", "#A855F7", "#F472B6"],
            template="plotly_dark", title="Missing Values by Column"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found.")
    st.write("Duplicate rows:", int(df.duplicated().sum()))

with tab3:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()
    chart_type = st.selectbox(
        "Select Visualization",
        ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
    )
    if chart_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Numerical Column", numeric_cols)
            fig = px.histogram(df, x=col, template="plotly_dark",
                               color_discrete_sequence=["#A855F7"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available.")
    elif chart_type == "Bar Chart":
        col = st.selectbox("Category Column", all_cols)
        counts = (df[col].fillna("Missing").astype(str).value_counts().head(20)
                  .rename_axis("Category").reset_index(name="Count"))
        fig = px.bar(counts, x="Category", y="Count", color="Count",
                     template="plotly_dark",
                     color_continuous_scale=["#22D3EE", "#A855F7"])
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X Axis", numeric_cols)
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y Axis", y_options)
            fig = px.scatter(df, x=x_col, y=y_col, template="plotly_dark",
                             color_discrete_sequence=["#22D3EE"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select a dataset with two numerical columns.")
    elif chart_type == "Box Plot":
        if numeric_cols:
            col = st.selectbox("Select Column", numeric_cols)
            fig = px.box(df, y=col, points="outliers", template="plotly_dark",
                         color_discrete_sequence=["#F472B6"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available.")
    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least two numerical columns are required.")

def make_profile(data):
    numeric = data.select_dtypes(include=np.number)
    categorical = data.select_dtypes(include=["object", "category", "bool", "string"])
    numeric_stats = {}
    for col in numeric.columns[:20]:
        s = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not s.empty:
            numeric_stats[str(col)] = {
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "std": round(float(s.std()), 3) if len(s) > 1 else 0
            }
    category_stats = {}
    for col in categorical.columns[:15]:
        s = data[col].dropna().astype(str)
        category_stats[str(col)] = {
            "unique": int(s.nunique()),
            "top_values": {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        }
    return {
        "rows": len(data), "columns": len(data.columns),
        "column_names": [str(c) for c in data.columns],
        "dtypes": {str(k): str(v) for k, v in data.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "numeric_statistics": numeric_stats,
        "categorical_statistics": category_stats
    }

with tab4:
    st.subheader("🤖 Multi-Agent Data Intelligence")
    question = st.text_area(
        "What should your AI team analyze?",
        value=("Analyze the dataset, identify important patterns, explain data quality "
               "issues, and recommend three actionable business improvements.")
    )
    with st.expander("Preview dataset statistics"):
        st.json(make_profile(df))
    run = st.button("🚀 RUN AI ANALYST TEAM", type="primary", use_container_width=True)

    if run:
        if not api_key:
            st.error("Gemini API key missing. Add GOOGLE_API_KEY to your .env file and restart Streamlit.")
        elif not model_name or not question.strip():
            st.error("Enter a model ID and analysis question.")
        else:
            try:
                with st.spinner("🧠 The three AI agents are working..."):
                    llm = LLM(model=model_name, api_key=api_key,
                              temperature=0.2, max_tokens=4096)
                    analyst = Agent(
                        role="Senior Data Analyst",
                        goal="Analyze verified dataset statistics accurately.",
                        backstory=("You are a meticulous analyst. Never invent numerical "
                                   "values or claim causation without evidence."),
                        llm=llm, verbose=False, allow_delegation=False
                    )
                    insight = Agent(
                        role="Business Intelligence Analyst",
                        goal="Identify useful insights and opportunities.",
                        backstory=("You interpret patterns, distinguish facts from hypotheses, "
                                   "and recommend further analysis where needed."),
                        llm=llm, verbose=False, allow_delegation=False
                    )
                    writer = Agent(
                        role="Executive Report Writer",
                        goal="Produce a professional actionable report.",
                        backstory="You create clear business reports with recommendations and limitations.",
                        llm=llm, verbose=False, allow_delegation=False
                    )
                    profile = json.dumps(make_profile(df), ensure_ascii=False)
                    task1 = Task(
                        description=f"""Analyze this verified dataset profile:
{profile}
User question: {question}
Explain dataset structure, data quality, numerical statistics, categorical patterns,
and limitations. Use only supplied facts. Do not invent statistics or causation.""",
                        expected_output="Evidence-based analysis with findings and limitations.",
                        agent=analyst
                    )
                    task2 = Task(
                        description=f"""Review the prior analysis.
User question: {question}
Identify business implications, opportunities, risks, and follow-up analyses.
Distinguish measured facts from hypotheses.""",
                        expected_output="Prioritized insights and recommendations.",
                        agent=insight, context=[task1]
                    )
                    task3 = Task(
                        description="""Write the final Markdown report with sections:
1. Executive Summary
2. Dataset Overview
3. Key Findings
4. Data Quality and Limitations
5. Business Opportunities and Risks
6. Recommendations
7. Suggested Next Analyses
Use prior results. Do not invent measurements.""",
                        expected_output="A complete professional Markdown report.",
                        agent=writer, context=[task1, task2]
                    )
                    crew = Crew(agents=[analyst, insight, writer], tasks=[task1, task2, task3],
                                process=Process.sequential, verbose=False)
                    result = crew.kickoff()
                    st.session_state["report"] = str(result)
                    st.session_state["report_question"] = question
                st.success("AI analysis completed successfully!")
            except Exception as e:
                message = str(e)
                lower = message.lower()
                if "429" in message or "quota" in lower or "resource_exhausted" in lower:
                    st.error("Gemini quota exceeded. Check AI Studio usage and limits; wait for reset or use an available model with quota.")
                elif "api key" in lower or "403" in message or "permission_denied" in lower:
                    st.error("Gemini authentication failed. Check your API key and API access.")
                else:
                    st.error("The AI workflow failed. Technical details:")
                    st.code(message)

    if st.session_state.get("report"):
        st.markdown("---")
        st.subheader("📝 AI Business Intelligence Report")
        st.caption("Question: " + st.session_state.get("report_question", ""))
        st.markdown(st.session_state["report"])
        st.download_button(
            "⬇️ Download AI Report",
            data=st.session_state["report"],
            file_name="nexus_ai_report.md",
            mime="text/markdown"
        )

st.markdown("---")
st.markdown("<center>⚡ NEXUS AI | CrewAI + Google Gemini | Advanced Multi-Agent Analytics</center>",
            unsafe_allow_html=True)
