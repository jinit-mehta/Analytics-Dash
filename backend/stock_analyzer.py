from alpha_vantage.timeseries import TimeSeries
from transformers import pipeline
import torch
import pymc as pm
import numpy as np
from market_context import get_market_context

# Initialize Alpha Vantage and Mistral-7B
ts = TimeSeries(key="AN0PSJSI22TGHNW8", output_format="pandas")
stock_analyzer = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def fetch_stock_data(ticker):
    try:
        data, _ = ts.get_daily(symbol=ticker, outputsize="compact")
        prices = data["4. close"].values
        return prices[-30:].tolist()  # Last 30 days
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def analyze_stock(ticker, document_text=""):
    # Fetch stock data and market context
    stock_prices = fetch_stock_data(ticker)
    market_context = get_market_context(f"{ticker} stock news")
    
    # Prepare prompt
    prompt = (
        f"Analyze the stock {ticker} based on the following:\n"
        f"Recent 30-day closing prices: {stock_prices}\n"
        f"Market context: {market_context}\n"
        f"Document content (if any): {document_text}\n"
        "Identify 2 trends, 2 risks, and 1 investment opportunity."
    )
    
    # Generate analysis
    result = stock_analyzer(prompt, max_length=500, num_return_sequences=1)
    analysis = result[0]["generated_text"]
    
    # Calculate confidence score
    confidence = calculate_confidence(stock_prices)
    
    return {
        "ticker": ticker,
        "analysis": analysis,
        "stock_prices": stock_prices,
        "confidence": confidence
    }

def calculate_confidence(prices):
    if isinstance(prices, str):  # Error case
        return 0.0
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=np.mean(prices), sigma=np.std(prices))
        sigma = pm.HalfNormal("sigma", sigma=10)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=prices)
        trace = pm.sample(1000, tune=500, return_inferencedata=False)
    return float(np.mean(trace["mu"]) / np.max(prices))  # Simplified confidence metric