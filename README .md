# 📊 AI Data Analyst (Groq)

An interactive Streamlit web application that allows you to upload CSV
files and analyze them using natural language powered by Groq LLMs.

## 🚀 Features

-   Upload any CSV file
-   Ask questions in plain English
-   Filter data (e.g. Master's degree holders)
-   Generate plots (histograms, bar charts, etc.)
-   View dataset preview
-   Works with Groq LLM (`llama-3.3-70b-versatile`)

## 🛠 Tech Stack

-   Python
-   Streamlit
-   Pandas
-   Matplotlib
-   LangChain Experimental
-   Groq API

## 📂 Project Structure

    sample/
     ├── ai.py
     ├── .env
     └── requirements.txt

## 🔑 Setup

1.  Create a Groq API key
2.  Add it to `.env` file:

    GROQ_API_KEY=your_api_key_here

3.  Install dependencies

    pip install -r requirements.txt

4.  Run the app

    streamlit run ai.py

## 🧪 Example Queries

-   How many rows are there?
-   How many columns are there?
-   Provide data of whose master's is completed
-   Plot histogram of prevailing_wage

## 📌 Author

Vaibhav Chaudhari\
Email: chaudharivaibhav471@gmail.com
