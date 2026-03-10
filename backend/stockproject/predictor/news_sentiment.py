import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import concurrent.futures

# NewsAPI configuration
NEWS_API_KEY = "144036cfedee4e678875c0e2ea5bd16c"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# GNews API configuration
GNEWS_API_KEY = "3f95c7a0d831aee6463a767c85c35739"
GNEWS_API_ENDPOINT = "https://gnews.io/api/v4/search"

# Stock/Index name mappings for better news search
STOCK_NEWS_QUERIES = {
    "RELIANCE.NS": ["Reliance Industries", "Mukesh Ambani Reliance"],
    "HDFCBANK.NS": ["HDFC Bank", "HDFC Bank India"],
    "AXISBANK.NS": ["Axis Bank India", "Axis Bank"],
    "SBIN.NS": ["State Bank of India", "SBI Bank"],
    "INFY.NS": ["Infosys", "Infosys Limited"],
    "TCS.NS": ["TCS", "Tata Consultancy Services"],
    "ICICIBANK.NS": ["ICICI Bank", "ICICI Bank India"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank"],
    "ADANIPORTS.NS": ["Adani Ports", "Adani Ports SEZ"],
    "ADANIENT.NS": ["Adani Enterprises", "Gautam Adani"],
    "BAJFINANCE.NS": ["Bajaj Finance", "Bajaj Finserv"],
    "BHARTIARTL.NS": ["Bharti Airtel", "Airtel India"],
    "ONGC.NS": ["ONGC", "Oil Natural Gas Corporation India"],
    "^NSEI": ["Nifty 50", "NSE India Nifty"],
    "^NSEBANK": ["Bank Nifty", "Nifty Bank Index"],
    "^NSEMDCP50": ["Nifty Midcap", "NSE Midcap India"],
    "^CNXAUTO": ["Nifty Auto", "Indian Auto Stocks"],
}

def fetch_news_articles_newsapi(query, from_date=None, to_date=None, language="en", page_size=10):
    """Fetch from NewsAPI with timeout and error handling"""
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": language,
        "pageSize": min(10, page_size),
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
    }

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"[DEBUG] NewsAPI: {len(articles)} articles for '{query}'")
            return articles
        else:
            print(f"[WARN] NewsAPI returned {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        print(f"[WARN] NewsAPI timeout for '{query}'")
        return []
    except Exception as e:
        print(f"[ERROR] NewsAPI: {e}")
        return []

def fetch_news_articles_gnews(query, from_date=None, to_date=None, language="en", max_results=10):
    """Fetch from GNews with timeout and error handling"""
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT00:00:00Z')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%dT23:59:59Z')

    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": language,
        "max": min(10, max_results),
        "from": from_date,
        "to": to_date,
    }

    try:
        response = requests.get(GNEWS_API_ENDPOINT, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"[DEBUG] GNews: {len(articles)} articles for '{query}'")
            return articles
        else:
            print(f"[WARN] GNews returned {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        print(f"[WARN] GNews timeout for '{query}'")
        return []
    except Exception as e:
        print(f"[ERROR] GNews: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not text or len(text.strip()) < 10:
        return 0.0
    try:
        # Limit text length for performance
        text = text[:1000]
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"[ERROR] Sentiment analysis: {e}")
        return 0.0

def get_average_sentiment(articles):
    """Calculate average sentiment from articles"""
    if not articles:
        return 0.0
    
    sentiments = []
    for article in articles[:20]:  # Limit to 20 articles for performance
        # Try multiple fields for content
        content = (
            article.get("content") or 
            article.get("description") or 
            article.get("title") or 
            ""
        )
        if content and len(content) > 20:
            sentiment = analyze_sentiment(content)
            if sentiment != 0.0:  # Only count non-neutral
                sentiments.append(sentiment)
    
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        print(f"[DEBUG] Average sentiment from {len(sentiments)} articles: {avg:.3f}")
        return avg
    return 0.0

def fetch_news_articles(query, from_date=None, to_date=None, language="en", page_size=20):
    """
    Fetch news articles with improved query building and parallel fetching
    """
    # Get better search queries for known symbols
    search_queries = STOCK_NEWS_QUERIES.get(query, [query])
    
    # Clean any remaining special chars
    clean_queries = []
    for q in search_queries:
        clean_q = q.replace('.NS', '').replace('^', '').replace('_', ' ')
        if clean_q:
            clean_queries.append(clean_q)
    
    if not clean_queries:
        clean_queries = [query.replace('.NS', '').replace('^', '')]
    
    print(f"[DEBUG] Searching news for: {clean_queries}")
    
    all_articles = []
    
    # Fetch from both APIs in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for q in clean_queries[:2]:  # Limit to 2 queries
            futures.append(executor.submit(fetch_news_articles_newsapi, q, from_date, to_date, language, 10))
            futures.append(executor.submit(fetch_news_articles_gnews, q, from_date, to_date, language, 10))
        
        for future in concurrent.futures.as_completed(futures, timeout=10):
            try:
                articles = future.result()
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                print(f"[WARN] Future failed: {e}")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url") or article.get("link")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
        elif not url:
            unique_articles.append(article)
    
    print(f"[DEBUG] Total unique articles: {len(unique_articles)}")
    return unique_articles[:page_size]
