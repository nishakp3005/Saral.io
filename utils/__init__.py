"""
Summarizer utilities package.

This package contains modules for web scraping and text summarization.
"""

# Import key functionality to make them available at the package level
try:
    from .scraper import ArticleScraper
except ImportError:
    pass

try:
    from .summarizer import ArticleSummarizer
except ImportError:
    pass

# Package version
__version__ = '0.1.0'