import streamlit as st
import time
import os
import nltk
from utils.scraper import ArticleScraper
from utils.summarizer import ArticleSummarizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Set up cache directories
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/summary_cache", exist_ok=True)

# Initialize scraper and summarizer - use optimized models
scraper = ArticleScraper(cache_dir="data/cache", summary_cache_dir="data/summary_cache")
summarizer = ArticleSummarizer(default_model="pegasus-cnn", cache_dir="data/summary_cache")

# Page configuration
st.set_page_config(
    page_title="Saral.io",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin: 0.5rem 0;
    }
    .article-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.25rem;
    }
    .metadata {
        font-size: 0.9rem;
        color: #616161;
        margin-bottom: 0.5rem;
    }
    .summary-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #212121;
    }
    .keyword-tag {
        background-color: #E3F2FD;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border: none;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# About section in sidebar
st.sidebar.markdown('<div class="subheader">About</div>', unsafe_allow_html=True)
st.sidebar.info(
    "Saral.io is an advanced news article summarizer that leverages AI to provide concise, "
    "insightful summaries of any online article. It offers multiple summarization methods, "
    "keyword extraction, sentiment analysis, and more.\n\n"
    "Simply paste any news URL to get started."
)
st.sidebar.markdown("---")

# Summarization options
st.sidebar.markdown('<div class="subheader">Summarization Options</div>', unsafe_allow_html=True)

# Add performance mode option
performance_mode = st.sidebar.radio(
    "Performance Mode",
    ["Balanced", "High Quality", "Fast"],
    index=0,
    help="Balanced: Good balance of speed and quality\nHigh Quality: Better summaries but slower\nFast: Quick summaries with less accuracy"
)

summarization_mode = st.sidebar.radio(
    "Summarization Mode",
    ["Auto", "Extractive", "Abstractive"],
    help="Auto: Choose best method based on article length\nExtractive: Select key sentences from the article\nAbstractive: Generate new summary text"
)

# Model selection (only show if abstractive is selected)
model_name = None
if summarization_mode == "Abstractive":
    # Select model options based on performance mode
    if performance_mode == "Fast":
        model_options = ["distilbart", "t5-small"]
    elif performance_mode == "High Quality":
        model_options = ["pegasus-cnn", "bart-cnn"]
    else:  # Balanced
        model_options = list(ArticleSummarizer.MODELS.keys())
    
    model_name = st.sidebar.selectbox(
        "Model",
        model_options,
        index=0,
        help="Select the AI model for generating the summary"
    )

# Summary length options - adjusted based on performance mode
if performance_mode == "Fast":
    max_length_default = 200
    min_length_default = 50
elif performance_mode == "High Quality":
    max_length_default = 250
    min_length_default = 100
else:  # Balanced
    max_length_default = 200
    min_length_default = 100

max_length = st.sidebar.slider(
    "Maximum Summary Length", 
    min_value=50, 
    max_value=1000, 
    value=max_length_default,
    help="Maximum number of words in the summary"
)

min_length = st.sidebar.slider(
    "Minimum Summary Length", 
    min_value=10, 
    max_value=300, 
    value=min_length_default,
    help="Minimum number of words in the summary"
)

# Extractive ratio (only for extractive mode)
ratio = 0.3
if summarization_mode == "Extractive":
    ratio = st.sidebar.slider(
        "Extraction Ratio", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.3, 
        step=0.05,
        help="Proportion of sentences to keep from original article"
    )

# Additional features
st.sidebar.markdown('<div class="subheader">Additional Features</div>', unsafe_allow_html=True)

include_keywords = st.sidebar.checkbox(
    "Extract Keywords",
    value=True,
    help="Identify and display key topics from the article"
)

include_reading_time = st.sidebar.checkbox(
    "Show Reading Time",
    value=True,
    help="Calculate and display the estimated reading time"
)

include_sentiment = st.sidebar.checkbox(
    "Analyze Sentiment",
    value=True if performance_mode != "Fast" else False,
    help="Determine if the article has a positive, negative or neutral tone"
)

use_cache = st.sidebar.checkbox(
    "Use Cache",
    value=True,
    help="Speed up processing by using cached results"
)



# Main content
st.markdown('<div class="main-header">📰 Saral.io News Article Summarizer</div>', unsafe_allow_html=True)
st.markdown(
    "Enter the URL of any news article to get a concise, intelligent summary. "
    "Saral uses advanced AI to extract key information, saving you time while keeping you informed."
)

# URL input
url = st.text_input("Enter Article URL", placeholder="https://example.com/news-article")

# Process button
col1, col2 = st.columns([1, 5])
process_button = col1.button("Summarize", type="primary", use_container_width=True)
    
# Show example URLs
with st.expander("Need examples? Try these URLs"):
    example_urls = [
        "https://www.bbc.com/news/articles/ckgxk40ndk1o",
        "https://www.theguardian.com/fashion/2025/apr/20/true-blue-why-the-chore-jacket-just-wont-quit",
        "https://www.nytimes.com/2025/04/23/business/china-tariffs-robots-automation.html",
        "https://edition.cnn.com/2025/04/22/business/trump-china-trade-war-reduction-hnk-intl/index.html",
        "https://www.reuters.com/world/india/top-indian-funds-bet-domestic-sectors-lead-market-rebound-amid-global-jitters-2025-04-23/",
        "https://www.aljazeera.com/news/liveblog/2025/4/23/live-israel-attacks-childrens-hospital-in-gaza-polio-campaign-at-a-halt"
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        for i in range(0, len(example_urls), 3):
            if st.button(example_urls[i].split('/')[2], key=example_urls[i], use_container_width=True):
                url = example_urls[i]
                process_button = True
    
    with col2:
        for i in range(1, len(example_urls), 3):
            if st.button(example_urls[i].split('/')[2], key=example_urls[i], use_container_width=True):
                url = example_urls[i]
                process_button = True
    
    with col3:
        for i in range(2, len(example_urls), 3):
            if st.button(example_urls[i].split('/')[2], key=example_urls[i], use_container_width=True):
                url = example_urls[i]
                process_button = True

# Process the URL
if process_button and url:
    # Prepare summary options based on performance mode
    summary_options = {
        'mode': summarization_mode.lower(),
        'model_name': model_name,
        'max_length': max_length,
        'min_length': min_length,
        'ratio': ratio,
        'keywords': include_keywords,
        'reading_time': include_reading_time,
        'sentiment': include_sentiment
    }
    
    with st.spinner("Processing article..."):
        # Show detailed progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Scraping
        status_text.text("Scraping article content...")
        progress_bar.progress(10)
        
        # Skip artificial delay in Fast mode
        if performance_mode != "Fast":
            time.sleep(0.3)
        
        article_data = scraper.scrape_article(url, use_cache=use_cache, generate_summary=True, summary_options=summary_options)
        
        progress_bar.progress(50)
        status_text.text("Generating summary...")
        
        # Skip artificial delay in Fast mode
        if performance_mode != "Fast":
            time.sleep(0.3)
        
        if article_data.get('success', False):
            progress_bar.progress(100)
            status_text.text("Done!")
            
            # Skip artificial delay in Fast mode
            if performance_mode != "Fast":
                time.sleep(0.2)
            
            # Clear status elements
            status_text.empty()
            progress_bar.empty()
            
            # Extract data from response
            title = article_data.get('title', 'No Title')
            text = article_data.get('text', '')
            summary = article_data.get('summary', {}).get('summary', 'Summary not available')
            summary_method = article_data.get('summary', {}).get('method', 'extractive')
            keywords = article_data.get('summary', {}).get('keywords', [])
            reading_time = article_data.get('summary', {}).get('reading_time', 0)
            sentiment = article_data.get('summary', {}).get('sentiment', None)
            authors = article_data.get('authors', [])
            publish_date = article_data.get('publish_date', None)
            source = article_data.get('source', 'Unknown')
            markdown = article_data.get('markdown', '')
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Article Details", "Full Text", "Markdown"])
            
            with tab1:
                st.markdown(f'<div class="article-title">{title}</div>', unsafe_allow_html=True)
                
                # Metadata row
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                
                with meta_col1:
                    st.markdown(f"**Source:** {source}")
                    
                with meta_col2:
                    if reading_time:
                        st.markdown(f"**Reading time:** {reading_time} min")
                        
                with meta_col3:
                    if summary_method:
                        method_name = summary_method.capitalize()
                        model_info = f" ({model_name})" if model_name and summary_method == "abstractive" else ""
                        st.markdown(f"**Summary method:** {method_name}{model_info}")
                
                # Summary content
                st.markdown("### Summary")
                st.markdown(f'<div class="summary-text">{summary}</div>', unsafe_allow_html=True)
                
                # Keywords section
                if keywords:
                    st.markdown("### Keywords")
                    keyword_html = " ".join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                
                # Sentiment analysis
                if sentiment:
                    st.markdown("### Sentiment Analysis")
                    sentiment_value = sentiment.get('sentiment', 'NEUTRAL')
                    confidence = sentiment.get('confidence', 0.5)
                    
                    # Determine emoji based on sentiment
                    emoji = "😊" if sentiment_value == 'POSITIVE' else "😔" if sentiment_value == 'NEGATIVE' else "😐"
                    
                    # Display sentiment with color coding
                    sentiment_color = "#2E7D32" if sentiment_value == 'POSITIVE' else "#C62828" if sentiment_value == 'NEGATIVE' else "#616161"
                    st.markdown(f'<span style="color:{sentiment_color};font-size:1.2rem;font-weight:bold;">{sentiment_value} {emoji}</span> <span style="color:#616161;">(Confidence: {confidence:.0%})</span>', unsafe_allow_html=True)
                
                # Download button
                safe_title = ''.join(c if c.isalnum() or c in [' ', '-'] else '_' for c in title[:30])
                st.download_button(
                    label="Download Summary as Markdown",
                    data=markdown,
                    file_name=f"summary-{safe_title}.md",
                    mime="text/markdown",
                )
            
            with tab2:
                st.markdown(f'<div class="article-title">{title}</div>', unsafe_allow_html=True)
                
                # Article metadata
                st.markdown("### Article Information")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown(f"**URL:** {url}")
                    st.markdown(f"**Source:** {source}")
                    if authors:
                        st.markdown(f"**Authors:** {', '.join(authors)}")
                
                with info_col2:
                    if publish_date:
                        st.markdown(f"**Published:** {publish_date}")
                    if reading_time:
                        st.markdown(f"**Estimated reading time:** {reading_time} minutes")
                    
                # Word count statistics
                st.markdown("### Article Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    word_count = len(text.split())
                    st.metric("Word Count", f"{word_count:,}")
                
                with stats_col2:
                    sentences = len(nltk.sent_tokenize(text))
                    st.metric("Sentences", f"{sentences:,}")
                
                with stats_col3:
                    compression = (1 - (len(summary.split()) / max(1, word_count))) * 100
                    st.metric("Compression", f"{compression:.0f}%")
            
            with tab3:
                st.markdown(f'<div class="article-title">{title}</div>', unsafe_allow_html=True)
                st.markdown("### Full Article Text")
                st.markdown(f'<div style="height:400px;overflow-y:scroll;padding:0.5rem;border:1px solid #e0e0e0;border-radius:4px;">{text}</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown("### Markdown Summary")
                st.code(markdown, language="markdown")
                st.download_button(
                    label="Download Markdown",
                    data=markdown,
                    file_name=f"summary-{safe_title}.md",
                    mime="text/markdown",
                )
        else:
            # Display error
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error processing article: {article_data.get('error', 'Unknown error')}")
            st.info("Make sure the URL is correct and points to a valid news article. Some sites may block web scraping.")

# Show debugging tools
if st.sidebar.checkbox("Show Debug Tools", False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Debug Options")
    
    if st.sidebar.button("Clear Cache"):
        try:
            cache_count = 0
            for file in os.listdir("data/cache"):
                os.remove(os.path.join("data/cache", file))
                cache_count += 1
            for file in os.listdir("data/summary_cache"):
                os.remove(os.path.join("data/summary_cache", file))
                cache_count += 1
            st.sidebar.success(f"Cache cleared successfully! ({cache_count} files removed)")
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {str(e)}")
    
    # Add model preloading option for faster first-time summaries
    if st.sidebar.button("Preload Models"):
        with st.sidebar.status("Loading models..."):
            try:
                # Load the most commonly used models
                summarizer.load_model("pegasus-cnn")
                summarizer.load_model("distilbart")
                st.sidebar.success("Models preloaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading models: {str(e)}")