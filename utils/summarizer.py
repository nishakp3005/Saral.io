# utils/summarizer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class ArticleSummarizer:
    def __init__(self):
        """Initialize the summarization models"""
        # For extractive summarization
        self.tokenizer = None
        self.model = None
        
        # For abstractive summarization
        self.abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Ensure nltk punkt is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def lazy_load_models(self):
        """Load models only when needed to save memory"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    def extractive_summarize(self, text, ratio=0.3):
        """
        Extractive summarization - selects most important sentences
        
        Args:
            text (str): Article text to summarize
            ratio (float): Proportion of sentences to keep (0.0-1.0)
            
        Returns:
            str: Summary text
        """
        if not text:
            return ""
            
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 3:
            return text  # No need to summarize very short texts
        
        # Calculate how many sentences to keep
        num_sentences = max(3, int(len(sentences) * ratio))
        
        # Simple approach: score sentences based on position
        # More sophisticated approaches would use embeddings and similarity
        
        # Give higher scores to sentences at the beginning
        position_scores = [1.0/(i+1) for i in range(len(sentences))]
        
        # Select top sentences and sort by original position
        ranked_indices = np.argsort(position_scores)[::-1][:num_sentences]
        selected_indices = sorted(ranked_indices)
        
        # Join selected sentences
        summary = ' '.join([sentences[i] for i in selected_indices])
        return summary
    
    def abstractive_summarize(self, text, max_length=150, min_length=50):
        """
        Abstractive summarization - generates new text as summary
        
        Args:
            text (str): Article text to summarize
            max_length (int): Maximum length of generated summary
            min_length (int): Minimum length of generated summary
            
        Returns:
            str: Summary text
        """
        if not text:
            return ""
            
        # Truncate input text if too long for BART
        max_input_length = 1024
        tokenized_text = self.abstractive_summarizer.tokenizer(text, truncation=True, max_length=max_input_length)
        decoded_text = self.abstractive_summarizer.tokenizer.decode(tokenized_text["input_ids"], skip_special_tokens=True)
        
        # Generate summary
        summary = self.abstractive_summarizer(decoded_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    
    def summarize(self, text, mode="extractive", max_length=150, min_length=50, ratio=0.3):
        """
        Generate summary based on selected mode
        
        Args:
            text (str): Article text to summarize
            mode (str): 'extractive' or 'abstractive'
            max_length (int): Maximum summary length (for abstractive)
            min_length (int): Minimum summary length (for abstractive)
            ratio (float): Proportion of text to keep (for extractive)
            
        Returns:
            str: Summary text
        """
        if mode.lower() == "extractive":
            return self.extractive_summarize(text, ratio)
        else:
            return self.abstractive_summarize(text, max_length, min_length)
    
    def format_to_markdown(self, title, summary, metadata=None):
        """
        Format the summary into markdown
        
        Args:
            title (str): Article title
            summary (str): Generated summary
            metadata (dict): Additional article metadata
            
        Returns:
            str: Formatted markdown text
        """
        md = f"# {title}\n\n"
        
        if metadata:
            if 'authors' in metadata and metadata['authors']:
                md += f"**Authors:** {', '.join(metadata['authors'])}\n\n"
            
            if 'publish_date' in metadata and metadata['publish_date']:
                md += f"**Published:** {metadata['publish_date']}\n\n"
            
            if 'keywords' in metadata and metadata['keywords']:
                md += f"**Keywords:** {', '.join(metadata['keywords'])}\n\n"
        
        md += f"## Summary\n\n{summary}\n\n"
        
        # Add source attribution
        if 'url' in metadata and metadata['url']:
            md += f"*Source: [{metadata['url']}]({metadata['url']})*\n"
        
        return md