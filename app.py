# app.py - Main Streamlit application

import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import re
import pandas as pd
import time
from newspaper import Article

# Try different import approaches for transformers
try:
    from transformers import pipeline
except ImportError:
    st.error("Could not import pipeline from transformers. Let's try a different approach.")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Define our own pipeline function
        def summarization_pipeline(text, max_length=150, min_length=50):
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return [{'summary_text': summary}]
    except ImportError:
        st.error("Could not import necessary transformers modules. Please install the latest version.")
        st.code("pip install -U transformers torch")
        st.stop()

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

# Simple extractive summarization function that doesn't rely on transformers
def simple_extractive_summarize(text, num_sentences=5):
    try:
        sentences = nltk.sent_tokenize(text)
        # Simple algorithm: take first few sentences as summary
        # This is a very basic approach - more sophisticated would weigh sentence importance
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
        article.nlp()  # This performs keywords extraction and summarization
        
        data = {
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'keywords': article.keywords,
            'summary': article.summary,
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
    max_token_limit = 1024
    if len(text.split()) > max_token_limit:
        text = " ".join(text.split()[:max_token_limit])
    
    try:
        if mode == "extractive":
            # Use simple extractive summarization as fallback
            return simple_extractive_summarize(text, num_sentences=max(3, int(max_length/30)))
        else:  # abstractive
            try:
                # Try to use the pipeline if available
                if 'pipeline' in globals():
                    summary = pipeline("summarization", model="facebook/bart-large-cnn")(
                        text, max_length=max_length, min_length=min_length, do_sample=False
                    )[0]['summary_text']
                else:
                    # Use our custom function if pipeline isn't available
                    summary = summarization_pipeline(
                        text, max_length=max_length, min_length=min_length
                    )[0]['summary_text']
                return summary
            except Exception as e:
                st.warning(f"Advanced summarization failed: {str(e)}. Using simple summarization instead.")
                return simple_extractive_summarize(text, num_sentences=max(3, int(max_length/30)))
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

# Function to convert summary to markdown format
def format_to_markdown(title, summary, authors=None, publish_date=None, keywords=None):
    md = f"# {title}\n\n"
    
    if authors:
        md += f"**Authors:** {', '.join(authors)}\n\n"
    
    if publish_date:
        md += f"**Published:** {publish_date.strftime('%Y-%m-%d') if publish_date else 'Unknown'}\n\n"
    
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
    ["Extractive", "Abstractive"],
    help="Extractive summarization selects key sentences from the article. Abstractive summarization generates new text that captures the essence of the article."
)

max_length = st.sidebar.slider(
    "Maximum Summary Length", 
    min_value=50, 
    max_value=500, 
    value=150,
    help="Maximum number of words in the summary"
)

min_length = st.sidebar.slider(
    "Minimum Summary Length", 
    min_value=30, 
    max_value=200, 
    value=50,
    help="Minimum number of words in the summary"
)

language = st.sidebar.selectbox(
    "Language",
    ["English", "Spanish", "French", "German", "Chinese"],
    index=0,
    help="Select the language of the article (Note: Best results with English)"
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
col1, col2 = st.columns([1, 5])
with col1:
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
                mode=summarization_mode.lower(),
                max_length=max_length,
                min_length=min_length
            )
            progress_bar.progress(100)
            
            # Display results
            st.subheader("Article Information")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Title:** {article_data['title']}")
                st.markdown(f"**Authors:** {', '.join(article_data['authors']) if article_data['authors'] else 'Unknown'}")
            with col2:
                st.markdown(f"**Published:** {article_data['publish_date'].strftime('%Y-%m-%d') if article_data['publish_date'] else 'Unknown'}")
                st.markdown(f"**Keywords:** {', '.join(article_data['keywords']) if article_data['keywords'] else 'None'}")
            
            # Display summary
            st.subheader("Summary")
            st.markdown(summary)
            
            # Markdown format
            markdown_summary = format_to_markdown(
                article_data['title'],
                summary,
                article_data['authors'],
                article_data['publish_date'],
                article_data['keywords']
            )
            
            # Download button for markdown
            st.download_button(
                label="Download Summary as Markdown",
                data=markdown_summary,
                file_name=f"summary-{article_data['title'][:30].replace(' ', '-')}.md",
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