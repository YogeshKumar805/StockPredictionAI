import numpy as np
import pandas as pd
from textblob import TextBlob
from utils import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_stochastic, fetch_stock_data
)

class StockRecommendationEngine:
    def __init__(self):
        self.weight_technical = 0.4
        self.weight_trend = 0.3
        self.weight_sentiment = 0.3
        
    def analyze_technical_indicators(self, df):
        """Analyze technical indicators for buy/sell signals"""
        signals = []
        
        # RSI Analysis
        rsi = calculate_rsi(df['Close'])
        if rsi.iloc[-1] < 30:
            signals.append(('RSI', 1, 'Oversold condition'))
        elif rsi.iloc[-1] > 70:
            signals.append(('RSI', -1, 'Overbought condition'))
            
        # MACD Analysis
        macd, signal, _ = calculate_macd(df['Close'])
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append(('MACD', 1, 'Bullish crossover'))
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append(('MACD', -1, 'Bearish crossover'))
            
        # Bollinger Bands Analysis
        upper, middle, lower = calculate_bollinger_bands(df['Close'])
        if df['Close'].iloc[-1] < lower.iloc[-1]:
            signals.append(('Bollinger', 1, 'Price below lower band'))
        elif df['Close'].iloc[-1] > upper.iloc[-1]:
            signals.append(('Bollinger', -1, 'Price above upper band'))
            
        return signals
    
    def analyze_price_trend(self, df):
        """Analyze price trends"""
        short_ma = df['Close'].rolling(window=20).mean()
        long_ma = df['Close'].rolling(window=50).mean()
        
        trend_score = 0
        trend_reasons = []
        
        # Moving Average Analysis
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            trend_score += 1
            trend_reasons.append('Short-term MA above long-term MA')
        else:
            trend_score -= 1
            trend_reasons.append('Short-term MA below long-term MA')
            
        # Price Momentum
        returns = df['Close'].pct_change()
        if returns.tail(5).mean() > 0:
            trend_score += 1
            trend_reasons.append('Positive price momentum')
        else:
            trend_score -= 1
            trend_reasons.append('Negative price momentum')
            
        return trend_score, trend_reasons
    
    def analyze_sentiment(self, news_articles):
        """Analyze news sentiment"""
        if not news_articles:
            return 0, ['No recent news available']
            
        sentiments = []
        for article in news_articles:
            text = article.get('title', '') + ' ' + article.get('summary', '')
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
            
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        if avg_sentiment > 0.2:
            return 1, ['Positive news sentiment']
        elif avg_sentiment < -0.2:
            return -1, ['Negative news sentiment']
        return 0, ['Neutral news sentiment']
    
    def get_recommendation(self, symbol, news_articles=None):
        """Generate comprehensive stock recommendation"""
        try:
            # Fetch historical data
            df = fetch_stock_data(symbol, period='6mo')
            
            # Technical Analysis
            technical_signals = self.analyze_technical_indicators(df)
            technical_score = np.mean([signal[1] for signal in technical_signals]) if technical_signals else 0
            
            # Trend Analysis
            trend_score, trend_reasons = self.analyze_price_trend(df)
            
            # Sentiment Analysis
            sentiment_score, sentiment_reasons = self.analyze_sentiment(news_articles)
            
            # Calculate weighted score
            final_score = (
                technical_score * self.weight_technical +
                trend_score * self.weight_trend +
                sentiment_score * self.weight_sentiment
            )
            
            # Generate recommendation
            if final_score > 0.3:
                recommendation = 'Buy'
                strength = 'Strong' if final_score > 0.6 else 'Moderate'
            elif final_score < -0.3:
                recommendation = 'Sell'
                strength = 'Strong' if final_score < -0.6 else 'Moderate'
            else:
                recommendation = 'Hold'
                strength = 'Neutral'
                
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'strength': strength,
                'score': final_score,
                'analysis': {
                    'technical': [{'indicator': s[0], 'signal': s[1], 'reason': s[2]} for s in technical_signals],
                    'trend': trend_reasons,
                    'sentiment': sentiment_reasons
                }
            }
            
        except Exception as e:
            raise Exception(f"Error generating recommendation for {symbol}: {str(e)}")
