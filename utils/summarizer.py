# utils/summarizer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib
import os
import json
from datetime import datetime
import platform  # For platform-based device detection
from functools import lru_cache

class ArticleSummarizer:
    """Advanced article summarization with multiple strategies and features"""
    
    # Optimized model selection with better defaults
    MODELS = {
        "pegasus-cnn": "google/pegasus-cnn_dailymail",  # Best for news articles
        "bart-cnn": "facebook/bart-large-cnn",          # Good balance of quality/speed
        "distilbart": "sshleifer/distilbart-cnn-12-6",  # Faster, smaller version
        "bart-xsum": "facebook/bart-large-xsum",        # Good for very concise summaries
        "t5-small": "t5-small"                          # Lightweight option
    }
    
    def __init__(self, default_model="distilbart", cache_dir=None, device=None):
        """Initialize the summarization models and resources"""
        self.default_model = default_model
        self.device = device or self._detect_device()
        self.models = {}
        self.tokenizers = {}
        self.summarizers = {}
        
        # Don't load any model on init - lazy load them when needed
        self.sentiment_analyzer = None
        
        # Set up caching
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Ensure nltk resources are available
        self._download_nltk_resources()
    
    def _detect_device(self):
        # Simple CPU/GPU detection - can be enhanced based on needs
        # transformers will use available devices automatically
        return "cpu"  # Default to CPU for simplicity and compatibility
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def load_model(self, model_name):
        """Load a specific model - only when needed"""
        if model_name not in self.models:
            model_path = self.MODELS.get(model_name, model_name)
            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
                self.summarizers[model_name] = pipeline(
                    "summarization", 
                    model=model_path, 
                    tokenizer=self.tokenizers[model_name],
                    # Let transformers handle device placement automatically
                )
                self.models[model_name] = True
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                return False
        return True
    
    def preprocess_text(self, text):
        """Clean and prepare text for summarization - improved for news articles"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common news article artifacts
        # Remove image captions often in brackets or parentheses
        text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)
        
        # Remove URL and social media references
        text = re.sub(r'http\S+|www\.\S+|@\w+', '', text)
        
        # Remove line breaks and multiple spaces
        text = re.sub(r'\n+|\r+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text, max_chunk_size=800):
        """Split text into chunks for processing long articles - optimized for speed"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Estimate token count - faster approximation
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def extractive_summarize(self, text, ratio=0.3, min_sentences=3):
        """Enhanced extractive summarization - optimized for speed"""
        if not text:
            return ""
            
        # Clean the text
        text = self.preprocess_text(text)
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= min_sentences:
            return text
            
        # Calculate how many sentences to keep
        num_sentences = max(min_sentences, int(len(sentences) * ratio))
        
        # For very long articles, use simpler scoring to avoid memory issues
        if len(sentences) > 300:
            # Simple position-based scoring - very fast
            position_scores = []
            for i, _ in enumerate(sentences):
                # Score the beginning and end sentences higher
                if i < len(sentences) * 0.2:  # First 20%
                    position_scores.append(0.8 + (0.2 * (1 - i / (len(sentences) * 0.2))))
                elif i > len(sentences) * 0.8:  # Last 20%
                    position_scores.append(0.6 + (0.2 * ((i - len(sentences) * 0.8) / (len(sentences) * 0.2))))
                else:  # Middle sentences
                    position_scores.append(0.5)
            
            # Select top sentences based on position
            ranked_indices = np.argsort(position_scores)[::-1][:num_sentences]
            selected_indices = sorted(ranked_indices)
            
        else:
            # Use TF-IDF for shorter texts
            try:
                # Create sentence embeddings using TF-IDF - more memory efficient config
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                sentence_vectors = vectorizer.fit_transform(sentences)
                
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(sentence_vectors)
                
                # Score sentences by their similarity to other sentences
                sentence_scores = np.sum(similarity_matrix, axis=1)
                
                # Factor in position (beginning and end sentences often more important)
                position_weight = 0.3
                max_pos_score = len(sentences)
                position_scores = [(max_pos_score - min(i, max_pos_score - i)) / max_pos_score 
                                for i in range(len(sentences))]
                
                # Combine scores
                combined_scores = (1 - position_weight) * sentence_scores + position_weight * np.array(position_scores)
                
                # Select top sentences and sort by original position
                ranked_indices = np.argsort(combined_scores)[::-1][:num_sentences]
                selected_indices = sorted(ranked_indices)
                
            except ValueError:
                # Fallback to position-based scoring if vectorization fails
                position_scores = [1.0/(i+1) + 0.5/(len(sentences)-i) for i in range(len(sentences))]
                ranked_indices = np.argsort(position_scores)[::-1][:num_sentences]
                selected_indices = sorted(ranked_indices)
        
        # Join selected sentences
        summary = ' '.join([sentences[i] for i in selected_indices])
        return summary
    
    @lru_cache(maxsize=32)
    def abstractive_summarize(self, text, model_name=None, max_length=150, min_length=50):
        """Enhanced abstractive summarization - optimized for quality and speed"""
        if not text:
            return ""
            
        # Use default model if none specified
        model_name = model_name or self.default_model
        if model_name not in self.models and not self.load_model(model_name):
            # If the requested model fails, try the default
            model_name = self.default_model
            if model_name not in self.models and not self.load_model(model_name):
                # If default fails too, use a lightweight fallback
                model_name = "distilbart"
                if model_name not in self.models and not self.load_model(model_name):
                    # Last resort: extractive
                    return self.extractive_summarize(text, ratio=0.25)
        
        summarizer = self.summarizers[model_name]
        
        # Clean the text
        text = self.preprocess_text(text)
        
        # Check if text is too long and needs chunking
        if len(text.split()) > 800:
            # Use smaller chunks for better processing
            chunks = self.chunk_text(text, max_chunk_size=800)
            chunk_summaries = []
            
            # Process each chunk (with length-based parameters)
            for chunk in chunks:
                try:
                    chunk_length = len(chunk.split())
                    chunk_max_len = min(max_length, max(50, chunk_length // 3))
                    chunk_min_len = min(min_length, max(30, chunk_length // 6))
                    
                    summary = summarizer(
                        chunk, 
                        max_length=chunk_max_len,
                        min_length=chunk_min_len,
                        do_sample=False, 
                        truncation=True
                    )[0]['summary_text']
                    chunk_summaries.append(summary)
                except Exception as e:
                    # If a chunk fails, add a simplified version
                    chunk_summaries.append(self.extractive_summarize(chunk, ratio=0.2))
            
            # For articles with many chunks, summarize recursively
            intermediate_summary = " ".join(chunk_summaries)
            if len(intermediate_summary.split()) > 1000:
                final_summary = self.abstractive_summarize(
                    intermediate_summary,
                    model_name=model_name,
                    max_length=max_length,
                    min_length=min_length
                )
                return final_summary
            else:
                # Final summary of the combined chunk summaries
                try:
                    final_summary = summarizer(
                        intermediate_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )[0]['summary_text']
                    return final_summary
                except:
                    return intermediate_summary
        else:
            # For shorter text, summarize directly
            try:
                summary = summarizer(
                    text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                return summary
            except Exception as e:
                # Fallback to extractive if model fails
                return self.extractive_summarize(text, ratio=0.25)
                
    # Other methods remain unchanged...
    
    def extract_keywords(self, text, top_n=7):
        """Extract key terms from the article - optimized for news relevance"""
        if not text or len(text) < 50:
            return []
            
        # Use summary for keyword extraction if text is too long
        # This is faster and often gives more relevant keywords
        if len(text) > 5000:
            # Get extractive summary for keyword extraction
            summary_text = self.extractive_summarize(text, ratio=0.2)
            if summary_text:
                text = summary_text
        
        # Clean the text
        text = self.preprocess_text(text)
        
        # Tokenize and filter out stopwords
        stop_words = set(stopwords.words('english'))
        # Add common news article words that aren't useful keywords
        stop_words.update(['said', 'according', 'reported', 'told', 'says', 'year', 'years', 'time', 'week'])
        
        words = [word.lower() for word in re.findall(r'\w+', text)
                if word.lower() not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Get unique keywords (avoid repeating similar terms)
        unique_keywords = []
        added_words = set()
        
        for word, _ in sorted_words:
            # Skip if we've added a similar word already
            if any(word[:4] == added[:4] for added in added_words):
                continue
            
            unique_keywords.append(word)
            added_words.add(word)
            
            if len(unique_keywords) >= top_n:
                break
        
        return unique_keywords
    
    def estimate_reading_time(self, text, wpm=250):
        """Estimate reading time for the article - fast calculation"""
        if not text:
            return 0
        # Quick word count approximation
        word_count = len(text.split())
        minutes = max(1, round(word_count / wpm))
        return minutes
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the article - using transformers pipeline"""
        try:
            # Initialize the sentiment analyzer once
            if not self.sentiment_analyzer:
                # Use a smaller, faster model for sentiment
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            
            # Only use the first part of the text for faster processing
            # For news, usually the beginning sets the tone
            if len(text) > 1000:
                # Use a combination of beginning and end of the text
                beginning = text[:1000]
                try:
                    ending = text[-500:] if len(text) > 1500 else ""
                    sample_text = beginning + " " + ending
                except:
                    sample_text = beginning
            else:
                sample_text = text
                
            # Get sentiment
            result = self.sentiment_analyzer(sample_text)[0]
            
            return {
                'sentiment': result['label'],
                'confidence': round(result['score'], 2)
            }
        except:
            # If sentiment analysis fails, return neutral
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def summarize(self, text, mode="auto", model_name=None, max_length=150, 
                 min_length=50, ratio=0.3, keywords=True, reading_time=True,
                 sentiment=False):
        """Generate comprehensive summary with optional features - optimized"""
        if not text or len(text.strip()) < 50:
            return {"summary": text, "error": "Text too short to summarize"}
        
        result = {}
        
        # Check cache first if enabled
        if self.cache_dir:
            cache_key = hashlib.md5(f"{text[:5000]}_{mode}_{model_name}_{max_length}_{min_length}_{ratio}".encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    pass  # Continue if cache read fails
        
        # Auto-select mode based on text length
        if mode == "auto":
            word_count = len(text.split())
            # For shorter articles, abstractive is better
            # For longer articles, extractive is faster and often better
            mode = "abstractive" if word_count < 1500 else "extractive"
        
        # Generate summary based on mode
        try:
            if mode.lower() == "extractive":
                result["summary"] = self.extractive_summarize(text, ratio)
                result["method"] = "extractive"
            else:  # abstractive
                # For abstractive, prefer distilbart for faster performance
                if not model_name:
                    model_name = self.default_model
                
                result["summary"] = self.abstractive_summarize(text, model_name, max_length, min_length)
                result["method"] = "abstractive"
                result["model"] = model_name
        except Exception as e:
            # Fallback to extra-ctive if abstractive fails
            result["summary"] = self.extractive_summarize(text, ratio)
            result["method"] = "extractive (fallback)"
            result["error"] = str(e)
        
        # Add optional features
        if reading_time:
            result["reading_time"] = self.estimate_reading_time(text)
            
        # Extract keywords from the summary if we have one, otherwise from text
        if keywords:
            if "summary" in result and len(result["summary"]) > 100:
                # Use the summary for keyword extraction - faster and often better results
                result["keywords"] = self.extract_keywords(result["summary"])
            else:
                result["keywords"] = self.extract_keywords(text)
            
        # Run sentiment on the summary not the full text - much faster
        if sentiment:
            if "summary" in result and result["summary"]:
                result["sentiment"] = self.analyze_sentiment(result["summary"])
            else:
                result["sentiment"] = self.analyze_sentiment(text[:1000])
        
        # Cache result if caching is enabled
        if self.cache_dir:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f)
            except:
                pass  # Continue if cache write fails
        
        return result
    
    def format_to_markdown(self, title, summary_data, metadata=None):
        """Format the summary results into rich markdown"""
        md = f"# {title}\n\n"
        
        # Add metadata if available
        if metadata:
            if 'authors' in metadata and metadata['authors']:
                authors = metadata['authors'] if isinstance(metadata['authors'], list) else [metadata['authors']]
                md += f"**Authors:** {', '.join(authors)}\n\n"
            
            if 'publish_date' in metadata and metadata['publish_date']:
                md += f"**Published:** {metadata['publish_date']}\n\n"
            
            if 'source' in metadata and metadata['source']:
                md += f"**Source:** {metadata['source']}\n\n"
        
        # Add reading time if available
        if 'reading_time' in summary_data:
            md += f"**Original Reading Time:** {summary_data['reading_time']} min\n\n"
        
        # Add summary
        md += f"## Summary\n\n{summary_data['summary']}\n\n"
        
        # Add keywords if available
        if 'keywords' in summary_data and summary_data['keywords']:
            md += f"## Keywords\n\n"
            md += ", ".join([f"`{keyword}`" for keyword in summary_data['keywords']])
            md += "\n\n"
        
        # Add sentiment if available
        if 'sentiment' in summary_data:
            sentiment = summary_data['sentiment']
            emoji = "😊" if sentiment['sentiment'] == 'POSITIVE' else "😐" if sentiment['sentiment'] == 'NEUTRAL' else "😔"
            md += f"**Sentiment:** {sentiment['sentiment']} {emoji} ({sentiment['confidence']:.0%} confidence)\n\n"
        
        # Add summary method
        if 'method' in summary_data:
            md += f"*Summary generated using {summary_data['method']} summarization"
            if 'model' in summary_data and summary_data['method'] == 'abstractive':
                md += f" with {summary_data['model']}"
            md += "*\n\n"
        
        # Add source attribution
        if metadata and 'url' in metadata and metadata['url']:
            md += f"*Source: [{metadata['url']}]({metadata['url']})*\n\n"
        
        # Add generation timestamp
        md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md