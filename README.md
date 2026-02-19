# Global News Topic Tracker

An AI-powered Streamlit web application that automatically fetches the latest news from around the world, identifies trending topics using machine learning, and summarizes them using the Google Gemini Large Language Model (LLM).


## Features
- **Automated News Aggregation** : Fetches and combines recent articles from multiple categories (Top News, World, Tech, Business).

- **AI-Powered Topic Clustering** : Uses sentence-transformers to create vector embeddings of news headlines and scikit-learn to cluster similar articles into distinct topics.

- **LLM-Generated Summaries** : For each topic cluster, it prompts the Google Gemini model to generate a concise title and a neutral, multi-sentence summary.

- **Interactive UI** : A user-friendly web interface built with Streamlit, allowing control over the number of articles to fetch and topics to identify.

- **Optimized Performance**: Caches expensive operations like fetching news and generating summaries to provide a fast and responsive experience on subsequent runs.

- **Secure API Key Handling**: Loads the Google AI API key securely from a .env file to keep it out of the source code.

## Tech Stack
Core Framework: Streamlit

Data Handling: Pandas

News Source: gnews

NLP & Clustering:

sentence-transformers

scikit-learn (KMeans)

AI Summarization: Google Generative AI (Gemini 1.5 Flash)

Environment Variables: python-dotenv



## Setup and Installation
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.8+

pip package manager

1. Clone the Repository
git clone repository
cd your-repo-name


2. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

3. Set Up Your API Key
The application requires a Google AI API Key to function.

Create a file named .env in the root directory of the project.

Add your API key to the .env file in the following format:

GOOGLE_API_KEY="AIzaSy...your...actual...api...key"

You can get your key from the Google AI Studio.

##🚀 How to Run the Application
Once you have completed the setup, run the following command in your terminal:

streamlit run app.py

Your web browser should automatically open a new tab with the running application.

Using the App
Use the sliders in the left sidebar to adjust the number of articles to fetch and the number of topics to identify.

Click the "🚀 Generate Trending Topics" button.

The application will display spinners while it works. Once complete, the main area will populate with expandable sections for each trending topic, including the AI-generated summary and a list of related articles.


## Streamlit interface:

![Chat Interface Preview](image1.png)

![Chat Interface Preview](image2.png)

