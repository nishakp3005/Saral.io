# utils/scraper.py
from newspaper import Article
import os
import json
import hashlib
from datetime import datetime
from .summarizer import ArticleSummarizer

class ArticleScraper:
    def __init__(self, cache_dir="data/cache", summary_cache_dir="data/summary_cache"):
        """Initialize the scraper with caching capability and summarization"""
        self.cache_dir = cache_dir
        self.summary_cache_dir = summary_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(summary_cache_dir, exist_ok=True)
        # Initialize the summarizer
        self.summarizer = ArticleSummarizer(cache_dir=summary_cache_dir)
    
    def _get_cache_filename(self, url):
        """Generate a cache filename based on URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def _is_cached(self, url):
        """Check if URL content is already cached"""
        cache_file = self._get_cache_filename(url)
        return os.path.exists(cache_file)
    
    def _get_from_cache(self, url):
        """Retrieve article data from cache"""
        cache_file = self._get_cache_filename(url)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, url, data):
        """Save article data to cache"""
        cache_file = self._get_cache_filename(url)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    
    def scrape_article(self, url, use_cache=True, generate_summary=False, summary_options=None):
        """
        Scrape article content from URL with caching and optional summarization
        
        Args:
            url (str): URL to scrape
            use_cache (bool): Whether to use cached data if available
            generate_summary (bool): Whether to generate summary of article
            summary_options (dict): Options for summarization (mode, model_name, etc.)
        
        Returns:
            dict: Article data including text, metadata, and optionally summary
        """
        # Check cache first if enabled
        if use_cache and self._is_cached(url):
            data = self._get_from_cache(url)
            # If summary requested but not in cached data, add it
            if generate_summary and 'summary' not in data:
                data = self._add_summary_to_data(data, summary_options)
                self._save_to_cache(url, data)  # Update cache with summary
            return data
        
        try:
            # Use newspaper3k for robust article extraction
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()  # Extract keywords and basic summary
            
            # Preprocess text
            cleaned_text = self.clean_text(article.text)
            
            # Calculate reading time
            reading_time = self.estimate_reading_time(cleaned_text)
            
            # Package the data
            data = {
                'title': article.title,
                'text': cleaned_text,
                'html': article.html,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'top_image': article.top_image,
                'keywords': article.keywords,
                'article_summary': article.summary,  # Original article summary
                'reading_time': reading_time,
                'url': url,
                'source': self._extract_source_name(url),
                'scrape_date': datetime.now().isoformat(),
                'success': True
            }
            
            # Generate summary if requested
            if generate_summary:
                data = self._add_summary_to_data(data, summary_options)
            
            # Cache the results
            self._save_to_cache(url, data)
            return data
            
        except Exception as e:
            error_data = {
                'url': url,
                'error': str(e),
                'success': False,
                'scrape_date': datetime.now().isoformat()
            }
            return error_data
    
    def clean_text(self, text):
        """Clean extracted text by removing extra whitespace"""
        if not text:
            return ""
        # Use the preprocessor from the summarizer for better cleaning
        return self.summarizer.preprocess_text(text)
    
    def _add_summary_to_data(self, data, summary_options=None):
        """Add summary to article data"""
        if not data.get('success', False) or not data.get('text'):
            return data
            
        options = summary_options or {}
        default_options = {
            'mode': 'auto',
            'model_name': None,
            'max_length': 150,
            'min_length': 50,
            'ratio': 0.3,
            'keywords': True,
            'reading_time': True,
            'sentiment': True
        }
        
        # Merge default options with provided options
        for key, value in default_options.items():
            if key not in options:
                options[key] = value
                
        # Generate summary
        summary_data = self.summarizer.summarize(data['text'], **options)
        
        # Add summary to data
        data['summary'] = summary_data
        
        # Generate markdown version
        metadata = {
            'authors': data.get('authors'),
            'publish_date': data.get('publish_date'),
            'url': data.get('url'),
            'source': data.get('source')
        }
        data['markdown'] = self.summarizer.format_to_markdown(
            data['title'], summary_data, metadata
        )
        
        return data
    
    def _extract_source_name(self, url):
        """Extract the source name from URL"""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Remove www. if present and get the main domain
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].capitalize()
        except:
            return None
    
    def estimate_reading_time(self, text, wpm=250):
        """Estimate reading time for the article"""
        if not text:
            return 0
        word_count = len(text.split())
        minutes = max(1, round(word_count / wpm))
        return minutes
    
    def generate_summary_for_cached(self, url, summary_options=None):
        """Generate summary for already cached article"""
        if not self._is_cached(url):
            return {"error": "Article not found in cache"}
        
        data = self._get_from_cache(url)
        data = self._add_summary_to_data(data, summary_options)
        self._save_to_cache(url, data)
        return data