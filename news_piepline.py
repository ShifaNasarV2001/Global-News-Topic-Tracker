#
# Global News Topic Tracker - Complete Pipeline
#
# This script fetches news, clusters articles into topics,
# and uses an LLM to summarize each topic.
#

# Step 1: Import necessary libraries
import os
import pandas as pd
from gnews import GNews
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import google.generativeai as genai

def setup_api_key():
    """Configures the Google AI API key from an environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    print("🔑 API key configured successfully.")

def fetch_top_headlines(max_results=100):
    """Fetches and combines news articles from various topics."""
    print("📡 Fetching top headlines...")
    google_news = GNews(language='en', country='US', max_results=max_results)
    
    # Fetch news from different categories for better topic diversity
    top_news = google_news.get_top_news()
    world_news = google_news.get_news_by_topic('WORLD')
    tech_news = google_news.get_news_by_topic('TECHNOLOGY')
    
    all_news = top_news + world_news + tech_news
    
    # Create a DataFrame and remove duplicate articles based on the title
    df = pd.DataFrame(all_news)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"📰 Fetched {len(df)} unique articles.")
    return df

def generate_embeddings(df):
    """Generates vector embeddings for article titles."""
    print("🧠 Generating text embeddings...")
    # Using a pre-trained model optimized for semantic search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Using article titles to generate embeddings
    embeddings = model.encode(df['title'].tolist(), show_progress_bar=True)
    return embeddings

def cluster_articles(embeddings, num_clusters):
    """Groups articles into clusters based on their embeddings."""
    print(f"📊 Clustering articles into {num_clusters} topics...")
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = clustering_model.fit_predict(embeddings)
    return labels

def summarize_topics_with_llm(df):
    """Generates a summary for each topic cluster using the Gemini LLM."""
    print("✍️  Summarizing topics with LLM...")
    summaries = {}
    model = genai.GenerativeModel('gemini-pro')

    # Loop through each unique cluster ID
    for cluster_id in sorted(df['cluster'].unique()):
        # Filter articles belonging to the current cluster
        cluster_articles_df = df[df['cluster'] == cluster_id]
        
        # Get the top 10 most relevant headlines for the prompt
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
            print(f"  - ✅ Successfully summarized cluster {cluster_id}")
        except Exception as e:
            summaries[cluster_id] = "**Topic Title:** Summary Failed\n**Summary:** Could not generate summary due to an API error."
            print(f"  - ❌ Failed to summarize cluster {cluster_id}. Error: {e}")
            
    return summaries

def main():
    """Main function to run the entire news tracking pipeline."""
    print("==============================================")
    print("       GLOBAL NEWS TOPIC TRACKER")
    print("==============================================")

    setup_api_key()
    
    news_df = fetch_top_headlines(max_results=100)
    
    if news_df.empty:
        print("No articles were found. Exiting.")
        return

    embeddings = generate_embeddings(news_df)
    
    # Heuristic for determining the number of topics: a tenth of the articles, but at least 5.
    num_topics = max(5, int(len(news_df) / 10))
    
    news_df['cluster'] = cluster_articles(embeddings, num_clusters=num_topics)
    
    topic_summaries = summarize_topics_with_llm(news_df)
    
    # --- Display Final Results ---
    print("\n\n==============================================")
    print("       🔥 LATEST TRENDING TOPICS 🔥")
    print(f"       (As of {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %I:%M %p IST')})")
    print("==============================================")
    
    for cluster_id, summary in topic_summaries.items():
        print(f"\n--- TRENDING TOPIC #{cluster_id + 1} ---\n")
        print(summary)

# This block ensures the main function runs when the script is executed
if __name__ == '__main__':
    main()
