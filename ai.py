import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# 1. ENV
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

PLOT_FILE = "temp_plot.png"

# 2. PAGE CONFIG
st.set_page_config("AI Data Analyst (Groq)", layout="wide")
st.title("📊 AI Data Analyst (Groq)")

# 3. SIDEBAR (RESTORED)

with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("---")

    st.markdown("### Example Questions")
    st.code("How many rows are there?")
    st.code("How many columns are there?")
    st.code("Provide data of whose master's is completed")
    st.code("Plot histogram of prevailing_wage")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        if os.path.exists(PLOT_FILE):
            os.remove(PLOT_FILE)
        st.experimental_rerun()

# 4. FILE UPLOAD

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if not uploaded_file:
    st.info("👆 Upload a CSV to start")
    st.stop()

df = pd.read_csv(uploaded_file)

with st.expander("🔍 Data Preview"):
    st.dataframe(df.head())
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

# 5. LLM (GROQ ONLY)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    # model = "llama-3.1-8b-instant",
    temperature=0
)

# 6. system prompt

SYSTEM_PROMPT = f"""
You are a strict Pandas Data Analyst.

CRITICAL DATA AWARENESS RULE:
The dataframe preview (df.head()) DOES NOT represent all values.
Whenever categories are needed (continent, education, status, etc),
you MUST compute from the FULL dataframe using:
df[column].unique()

Never guess categories from df.head().

---------------------------
MANDATORY RULES
---------------------------

1. ROW COUNT
Answer only:
Rows: <number>

2. COLUMN COUNT
Answer only:
Columns: <number>

3. COLUMN NAMES
Format:
Columns (<count>):
- col1
- col2

4. FILTERING / DATA DISPLAY
- Always filter using pandas
- ALWAYS apply .head(2)
- ALWAYS output using:
  df_filtered.head(2).to_markdown(index=False)
- If rows > 2, add line:
  Showing first 2 rows only.

5. EDUCATION FILTER
"master", "masters", "master's"
→ education_of_employee == "Master's"

6. PLOTS
- matplotlib only
- Save to '{PLOT_FILE}'
- Do not show plot
- End with:
  Plot generated successfully.

7. CATEGORICAL QUERIES
For questions like:
- list continents
- unique regions
- education types

Always compute:
df[column].unique()

Never rely on preview rows.

NO raw DataFrame output.
NO plain text tables.
"""


# 7. AGENT (CORRECT TYPE)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True,
    prefix=SYSTEM_PROMPT
)

# 8. CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 9. CHAT INPUT

prompt = st.chat_input("Ask a question about your data")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                if os.path.exists(PLOT_FILE):
                    os.remove(PLOT_FILE)
                plt.clf()

                result = agent.invoke(prompt)
                output = result["output"]

                if os.path.exists(PLOT_FILE):
                    st.image(PLOT_FILE)
                else:
                    st.markdown(output)

                st.session_state.messages.append(
                    {"role": "assistant", "content": output}
                )

            except Exception as e:
                st.error(f"❌ Agent Error: {e}")




#  the run command for streamlit is python -m streamlit run ai.py 
