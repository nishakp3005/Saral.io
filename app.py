# app.py - Simplified version without pandas dependency
import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import time
from newspaper import Article

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Summarizer.com",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple extractive summarization function
def simple_extractive_summarize(text, num_sentences=5):
    try:
        sentences = nltk.sent_tokenize(text)
        # Simple algorithm: take first few sentences as summary
        if len(sentences) <= num_sentences:
            return text
        
        return " ".join(sentences[:num_sentences])
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Function to scrape and process article
def process_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        try:
            article.nlp()  # This performs keywords extraction and summarization
        except:
            # If NLP fails, continue without keywords and auto-summary
            pass
        
        data = {
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'keywords': getattr(article, 'keywords', []),
            'summary': getattr(article, 'summary', ''),
            'success': True
        }
        return data
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Function to generate summary
def generate_summary(text, mode="extractive", max_length=150, min_length=50):
    if not text:
        return ""
    
    # Truncate text if it's too long
    words = text.split()
    max_token_limit = 1024
    if len(words) > max_token_limit:
        text = " ".join(words[:max_token_limit])
    
    # For now, we'll just use our simple extractive summarization
    # Adjust number of sentences based on desired length
    num_sentences = max(3, int(max_length/30))
    return simple_extractive_summarize(text, num_sentences=num_sentences)

# Function to convert summary to markdown format
def format_to_markdown(title, summary, authors=None, publish_date=None, keywords=None):
    md = f"# {title}\n\n"
    
    if authors:
        md += f"**Authors:** {', '.join(authors)}\n\n"
    
    if publish_date:
        md += f"**Published:** {publish_date.strftime('%Y-%m-%d') if hasattr(publish_date, 'strftime') else str(publish_date)}\n\n"
    
    if keywords:
        md += f"**Keywords:** {', '.join(keywords)}\n\n"
    
    md += f"## Summary\n\n{summary}\n\n"
    return md

# Sidebar
st.sidebar.title("Summarizer.com")
st.sidebar.markdown("📰 News Article Summarizer")

# Summarization options
st.sidebar.subheader("Summarization Options")
summarization_mode = st.sidebar.radio(
    "Select Summarization Mode",
    ["Extractive"],  # Simplified to just extractive for now
    help="Extractive summarization selects key sentences from the article."
)

max_length = st.sidebar.slider(
    "Maximum Summary Length", 
    min_value=50, 
    max_value=1000, 
    value=150,
    help="Maximum number of words in the summary"
)

# About section in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application helps you summarize news articles from any URL. "
    "Simply paste the URL of a news article, and the platform will scrape "
    "the content and generate a concise summary in markdown format."
)

# Main content
st.title("📰 News Article Summarizer")
st.markdown(
    "Enter the URL of any news article to get a concise summary. "
    "Perfect for quickly understanding complex news without reading the entire article."
)

# URL input
url = st.text_input("Enter Article URL", placeholder="https://example.com/news-article")

# Process button
process_button = st.button("Summarize", type="primary")

# Show example URLs
with st.expander("Need examples? Try these URLs"):
    example_urls = [
        "https://www.bbc.com/news/world-us-canada-67063546",
        "https://www.theguardian.com/technology/article/2023/oct/15/ai-artificial-intelligence-future",
        "https://www.nytimes.com/2023/10/14/climate/climate-change-solutions.html"
    ]
    for ex_url in example_urls:
        if st.button(ex_url, key=ex_url):
            url = ex_url
            process_button = True

# Process the URL
if process_button and url:
    with st.spinner("Processing article..."):
        # Show progress
        progress_bar = st.progress(0)
        for i in range(100):
            # Simulate progress while tasks are running
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Process the article
        article_data = process_article(url)
        
        if article_data['success']:
            # Generate summary based on selected mode
            progress_bar.progress(80)
            summary = generate_summary(
                article_data['text'], 
                mode="extractive",  # Simplified to just extractive
                max_length=max_length
            )
            progress_bar.progress(100)
            
            # Display results
            st.subheader("Article Information")
            
            st.markdown(f"**Title:** {article_data['title']}")
            st.markdown(f"**Authors:** {', '.join(article_data['authors']) if article_data['authors'] else 'Unknown'}")
            
            publish_date = article_data['publish_date']
            if publish_date:
                if hasattr(publish_date, 'strftime'):
                    formatted_date = publish_date.strftime('%Y-%m-%d')
                else:
                    formatted_date = str(publish_date)
                st.markdown(f"**Published:** {formatted_date}")
            
            keywords = article_data.get('keywords', [])
            if keywords:
                st.markdown(f"**Keywords:** {', '.join(keywords)}")
            
            # Display summary
            st.subheader("Summary")
            st.markdown(summary)
            
            # Markdown format
            markdown_summary = format_to_markdown(
                article_data['title'],
                summary,
                article_data['authors'],
                article_data['publish_date'],
                article_data.get('keywords', [])
            )
            
            # Download button for markdown
            safe_title = ''.join(c if c.isalnum() or c in [' ', '-'] else '_' for c in article_data['title'][:30])
            st.download_button(
                label="Download Summary as Markdown",
                data=markdown_summary,
                file_name=f"summary-{safe_title}.md",
                mime="text/markdown",
            )
            
        else:
            st.error(f"Error processing article: {article_data['error']}")
            st.info("Make sure the URL is correct and points to a valid news article.")

# Additional utility to show raw markdown
if st.sidebar.checkbox("Show Raw Markdown", False):
    if 'markdown_summary' in locals():
        st.subheader("Raw Markdown")
        st.code(markdown_summary)