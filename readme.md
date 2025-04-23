# Saral.io

A powerful web application designed to summarize news articles from any URL. Users can input the URL of a news article, and the platform will scrape the content, process it, and generate a concise summary in markdown format.

## Features

- **URL Input**: Enter any news article URL to process
- **Smart Web Scraping**: Extracts article content intelligently
- **Dual Summarization Modes**:
  - **Extractive**: Identifies and extracts the most important sentences
  - **Abstractive**: Generates a completely new summary that captures the essence
- **Customizable Summary Length**: Adjust the summary size to your preference
- **Markdown Formatting**: Summaries are formatted in clean markdown
- **Multilingual Support**: Works with articles in multiple languages
- **Caching**: Improves performance by caching previously scraped articles

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/summarizer.git
   cd summarizer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Requirements

- Python 3.8+
- Streamlit
- Transformers (Hugging Face)
- NLTK
- Newspaper3k
- Beautiful Soup 4
- Requests

## Usage

1. Enter the URL of a news article in the input field
2. Select your preferred summarization mode (Extractive or Abstractive)
3. Adjust the summary length using the sliders
4. Click "Summarize" to process the article
5. View the summary and download it as a markdown file if desired

## Project Structure

```
summarizer/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
├── .gitignore          
├── README.md            # Project documentation
├── utils/
│   ├── __init__.py
│   ├── scraper.py       # Web scraping functionality
│   └── summarizer.py    # Summarization models and functions
│
└── data/
    └── cache/           # For caching scraped articles
```

## License

MIT

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/) for NLP models
- [Newspaper3k](https://newspaper.readthedocs.io/) for article extraction
- [Streamlit](https://streamlit.io/) for the web interface