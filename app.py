import os
if os.path.exists("streamlit_secret") and not os.path.exists(".streamlit"):
    os.rename("streamlit_secret", ".streamlit")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
import os

st.set_page_config(page_title="Agentic AI for Tweet Analysis", layout="centered")
st.title("🤖 Agentic AI: Social Media Analyzer")
st.markdown("Upload a tweet CSV, enter your task in natural language, and let the AI agent do the rest.")

# Huggingface model for sentiment
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")

    # Get user task
    task = st.text_input("🧠 What do you want the agent to do?", 
                         "Analyze Beko dishwasher tweets, summarize sentiment, and show word cloud.")

    if st.button("🚀 Run Agent") and task:
        # Tool 1: Sentiment analysis
        def sentiment_tool(_: str):
            filtered = df[(df['Brand'].str.lower() == "beko") & (df['Category'].str.lower() == "dishwasher")].copy()
            filtered['Sentiment'] = filtered['Tweet'].apply(lambda x: sentiment_model(x)[0]['label'])
            global sentiment_counts
            sentiment_counts = filtered['Sentiment'].value_counts()
            return str(sentiment_counts.to_dict())

        # Tool 2: Summary
        def summary_tool(_: str):
            total = sentiment_counts.sum()
            pos = sentiment_counts.get('POSITIVE', 0)
            neg = sentiment_counts.get('NEGATIVE', 0)
            neu = sentiment_counts.get('NEUTRAL', 0)
            return f"Out of {total} tweets, {pos} are positive, {neg} negative, and {neu} neutral."

        # Tool 3: WordCloud
        def wordcloud_tool(_: str):
            text = " ".join(df[(df['Brand'].str.lower() == "beko") & (df['Category'].str.lower() == "dishwasher")]['Tweet'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            return "Word cloud shown above."

        # LangChain Agent
        tools = [
            Tool(name="SentimentAnalyzer", func=sentiment_tool, description="Analyze Beko dishwasher tweet sentiments."),
            Tool(name="SummaryGenerator", func=summary_tool, description="Summarize sentiment counts."),
            Tool(name="WordCloudGenerator", func=wordcloud_tool, description="Generate word cloud of tweet text.")
        ]

        llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.5)
        agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        with st.spinner("Agent is thinking..."):
            output = agent.invoke(task)
            st.success("Done!")
            st.write("### 📝 Agent Output:")
            st.write(output)
