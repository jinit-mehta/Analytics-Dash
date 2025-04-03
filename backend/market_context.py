from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="43e6f0c16dad4e5695a19daa9a501dbf")  # Get free key from newsapi.org

def get_market_context(query="latest financial news"):
    articles = newsapi.get_everything(q=query, language="en", page_size=5)
    context = "\n".join([article["title"] for article in articles["articles"]])
    return context if context else "No recent market context available."