# utils/scraper.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import os
import json
import hashlib
from datetime import datetime

class ArticleScraper:
    def __init__(self, cache_dir="data/cache"):
        """Initialize the scraper with caching capability"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
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
    
    def scrape_article(self, url, use_cache=True):
        """Scrape article content from URL with caching"""
        # Check cache first if enabled
        if use_cache and self._is_cached(url):
            return self._get_from_cache(url)
        
        try:
            # Use newspaper3k for robust article extraction
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()  # Extract keywords and basic summary
            
            # Package the data
            data = {
                'title': article.title,
                'text': article.text,
                'html': article.html,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'top_image': article.top_image,
                'keywords': article.keywords,
                'summary': article.summary,
                'url': url,
                'scrape_date': datetime.now().isoformat(),
                'success': True
            }
            
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
        # Remove multiple spaces and newlines
        cleaned = ' '.join(text.split())
        return cleaned