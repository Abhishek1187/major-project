import requests
from textblob import TextBlob
from datetime import datetime, timedelta

# NewsAPI configuration (working)
NEWS_API_KEY = "144036cfedee4e678875c0e2ea5bd16c"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

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

        response = requests.get(NEWS_API_ENDPOINT, params=params)
        if response.status_code == 200:
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
        else:
            print(f"[ERROR] NewsAPI: Failed to fetch news articles: {response.status_code} - {response.text}")
            break
    return all_articles[:page_size]

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

    # Broaden query by adding more synonyms or related keywords
    expanded_query = (
        f"{query} OR {query.replace('NIFTY', 'NSE')} OR {query.replace('BANK', 'BANKING')} "
        f"OR {query.replace('FINANCE', 'FINANCIAL')} OR {query.replace('AUTO', 'AUTOMOBILE')}"
    )

    # Fetch from NewsAPI (our primary and working source)
    articles = fetch_news_articles_newsapi(expanded_query, from_date, to_date, language, page_size)

    # Remove duplicates based on article URL if available
    seen_urls = set()
    unique_articles = []
    for article in articles:
        url = article.get("url") or article.get("link") or None
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
        elif not url:
            # If no URL, include anyway
            unique_articles.append(article)

    return unique_articles[:page_size]
