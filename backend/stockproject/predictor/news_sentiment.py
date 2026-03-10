import requests
from textblob import TextBlob
from datetime import datetime, timedelta

# NewsAPI configuration (working)
NEWS_API_KEY = "144036cfedee4e678875c0e2ea5bd16c"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# GNews API configuration (activated)
GNEWS_API_KEY = "3f95c7a0d831aee6463a767c85c35739"
GNEWS_API_ENDPOINT = "https://gnews.io/api/v4/search"

def fetch_news_articles_newsapi(query, from_date=None, to_date=None, language="en", page_size=100):
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')

    all_articles = []
    page = 1
    total_results = None
    while len(all_articles) < page_size:
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": language,
            "pageSize": min(20, page_size - len(all_articles)),
            "page": page,
            "from": from_date,
            "to": to_date,
        }

        try:
            response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[ERROR] NewsAPI: Exception occurred: {e}")
            # Disable NewsAPI by returning empty list on failure
            return []
            # break

        data = response.json()
        if total_results is None:
            total_results = data.get("totalResults", 0)
            print(f"[DEBUG] NewsAPI: Total results for query '{query}': {total_results}")
        articles = data.get("articles", [])
        print(f"[DEBUG] NewsAPI: Fetched {len(articles)} articles for query '{query}' page {page}")
        all_articles.extend(articles)
        if len(articles) < params["pageSize"]:
            # No more articles available
            break
        if len(all_articles) >= total_results:
            # All articles fetched
            break
        page += 1
    return all_articles[:page_size]

def fetch_news_articles_gnews(query, from_date=None, to_date=None, language="en", max_results=100):
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": language,
        "max": max_results,
        "from": from_date,
        "to": to_date,
    }
    print(f"[DEBUG] GNews: Request params: {params}")

    response = requests.get(GNEWS_API_ENDPOINT, params=params)
    if response.status_code == 200:
        data = response.json()
        total_results = data.get("totalArticles", 0)
        print(f"[DEBUG] GNews: Total results for query '{query}': {total_results}")
        articles = data.get("articles", [])
        print(f"[DEBUG] GNews: Fetched {len(articles)} articles for query '{query}'")
        return articles[:max_results]
    else:
        print(f"[ERROR] GNews: Failed to fetch news articles: {response.status_code} - {response.text}")
        return []

def analyze_sentiment(text):
    if not text or text.strip() == "":
        return 0.0
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        print(f"[DEBUG] Sentiment polarity for text: {polarity}")
        return polarity
    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return 0.0

def get_average_sentiment(articles):
    if not articles:
        return 0.0
    sentiments = []
    for article in articles:
        content = article.get("content") or article.get("description") or ""
        if content:
            sentiment = analyze_sentiment(content)
            print(f"[DEBUG] Article sentiment: {sentiment}")
            sentiments.append(sentiment)
    if sentiments:
        average = sum(sentiments) / len(sentiments)
        print(f"[DEBUG] Average sentiment: {average}")
        return average
    else:
        print("[DEBUG] No sentiments found, returning 0.0")
        return 0.0

def fetch_news_articles(query, from_date=None, to_date=None, language="en", page_size=100):
    # Increase date range to 30 days if not specified
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')

    # Clean query for GNews - remove special characters and simplify
    clean_query = query.replace('.NS', '').replace('^', '').replace('_', ' ')
    
    # Create different queries for different APIs
    newsapi_query = (
        f"{query} OR {query.replace('NIFTY', 'NSE')} OR {query.replace('BANK', 'BANKING')} "
        f"OR {query.replace('FINANCE', 'FINANCIAL')} OR {query.replace('AUTO', 'AUTOMOBILE')}"
    )
    
    # Simplified query for GNews (no OR operators for problematic symbols)
    gnews_query = clean_query

    all_articles = []

    # Fetch from NewsAPI
    articles_newsapi = fetch_news_articles_newsapi(newsapi_query, from_date, to_date, language, page_size)
    if articles_newsapi:
        all_articles.extend(articles_newsapi)

    # Fetch from GNews API with simplified query
    articles_gnews = fetch_news_articles_gnews(gnews_query, from_date, to_date, language, page_size)
    if articles_gnews:
        all_articles.extend(articles_gnews)

    # Remove duplicates based on article URL if available
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url") or article.get("link") or None
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
        elif not url:
            # If no URL, include anyway
            unique_articles.append(article)

    return unique_articles[:page_size]
