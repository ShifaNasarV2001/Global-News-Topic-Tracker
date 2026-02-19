# -*- coding: utf-8 -*-
"""
Global News Topic Tracker - Streamlit Web Application

This script creates an interactive web interface for the news topic
analysis pipeline. Users can adjust settings and view auto-summarized
news topics. The API key is hardcoded in this version.
"""

# Step 1: Import necessary libraries
import os
import pandas as pd
import streamlit as st
from gnews import GNews
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import google.generativeai as genai

# --- API CONFIGURATION ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Google AI API key.
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the API key at the start
try:
    # A quick check to ensure the placeholder is replaced
    if API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Please replace 'YOUR_API_KEY_HERE' with your actual API key in the script.", icon="🔑")
        st.stop()
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"❌ Error configuring API key: {e}")
    st.info("Please make sure you have set your API key correctly in the script.")
    st.stop()


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global News Topic Tracker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CORE FUNCTIONS (with Caching) ---

@st.cache_data(show_spinner="📡 Fetching news from around the world...")
def fetch_top_headlines(max_results=100):
    """Fetches and combines news articles from various topics."""
    google_news = GNews(language='en', country='US', max_results=max_results)
    top_news = google_news.get_top_news()
    world_news = google_news.get_news_by_topic('WORLD')
    tech_news = google_news.get_news_by_topic('TECHNOLOGY')
    business_news = google_news.get_news_by_topic('BUSINESS')
    
    all_news = top_news + world_news + tech_news + business_news
    
    df = pd.DataFrame(all_news)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    return df

@st.cache_data(show_spinner="🧠 Generating text embeddings...")
def generate_embeddings(_df):
    """Generates vector embeddings for article titles."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(_df['title'].tolist(), show_progress_bar=False)
    return embeddings

@st.cache_data(show_spinner="✍️ Summarizing topics with AI...")
def summarize_topics_with_llm(_df, num_topics):
    """Generates a summary for each topic cluster using the Gemini LLM."""
    summaries = {}
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    for cluster_id in sorted(_df['cluster'].unique()):
        cluster_articles_df = _df[_df['cluster'] == cluster_id]
        article_headlines = "- " + "\n- ".join(cluster_articles_df['title'].head(10))
        
        prompt = f"""
        You are a senior news editor. Based on the following list of news headlines, which all relate to the same topic, please perform these two tasks:

        1.  **Generate a Topic Title:** Create a short, descriptive title (3-5 words) that captures the main event.
        2.  **Write a Summary:** Provide a concise, neutral summary (2-3 sentences) explaining the core news story.

        Headlines:
        {article_headlines}

        Provide the output in this exact format:
        **Topic Title:** [Your Title Here]
        **Summary:** [Your Summary Here]
        """

        try:
            response = model.generate_content(prompt)
            summaries[cluster_id] = response.text
        except Exception as e:
            summaries[cluster_id] = f"**Topic Title:** Summary Failed\n**Summary:** Could not generate summary. Error: {str(e)}"
            
    return summaries

# --- STREAMLIT UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Controls
    st.subheader("Model Controls")
    article_count = st.slider(
        "Articles to Fetch (approx.)", 
        min_value=50, max_value=300, value=100, step=10,
        help="Number of recent articles to fetch for analysis. More articles provide better context but take longer."
    )
    
    num_topics = st.slider(
        "Number of Topics to Identify", 
        min_value=5, max_value=20, value=8, step=1,
        help="How many distinct news topics to cluster the articles into."
    )
    
    # Action Button
    st.markdown("---")
    run_button = st.button("🚀 Generate Trending Topics")
    st.markdown("---")
    st.info("🕒 Note: The first run can take a minute as models are downloaded and caches are built.")


# --- Main Content Area ---
st.title("📰 Global News Topic Tracker")
st.markdown("An AI-powered tool to automatically discover and summarize the top trending news stories right now.")

if not run_button:
    st.info("Adjust the settings in the sidebar and click 'Generate Trending Topics' to begin.")
    st.stop()

# --- ANALYSIS PIPELINE ---
# This block runs only when the button is pressed.

# 1. Fetch data
news_df = fetch_top_headlines(max_results=int(article_count / 4)) # GNews fetches per category

if news_df.empty:
    st.warning("Could not fetch any news articles. Please try again later.")
    st.stop()

# 2. Generate embeddings
embeddings = generate_embeddings(news_df)

# 3. Cluster articles
clustering_model = KMeans(n_clusters=num_topics, random_state=42, n_init='auto')
news_df['cluster'] = clustering_model.fit_predict(embeddings)

# 4. Summarize with LLM
topic_summaries = summarize_topics_with_llm(news_df, num_topics)

# 5. Display results
st.header(f" Top {num_topics} Trending Topics")
st.markdown(f"_Analysis based on {len(news_df)} unique articles._")

# Sort clusters by the number of articles in each, descending
sorted_clusters = news_df['cluster'].value_counts().index

for cluster_id in sorted_clusters:
    summary_text = topic_summaries.get(cluster_id, "Summary not available.")
    
    # Extract title from the summary markdown
    try:
        topic_title = summary_text.split("**Topic Title:**")[1].split("\n")[0].strip()
    except IndexError:
        topic_title = f"Topic #{cluster_id + 1}"

    with st.expander(f"**{topic_title}**", expanded=True):
        st.markdown(summary_text)
        
        st.markdown("---")
        st.subheader("Related Articles in this Topic")
        
        # Filter and display relevant articles
        cluster_articles_df = news_df[news_df['cluster'] == cluster_id]
        
        # Display a clean dataframe
        st.dataframe(
            cluster_articles_df[['title', 'publisher', 'published date']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "publisher": st.column_config.TextColumn("Publisher"),
                "published date": st.column_config.DatetimeColumn("Published", format="D MMM, h:mm a")
            }
        )

