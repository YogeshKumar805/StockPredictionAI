import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from textblob import TextBlob
import requests
import json
from models import StockPredictor
from utils import fetch_stock_data, prepare_data, calculate_metrics
from styles import apply_custom_styles, show_metric_card, show_footer
from recommendation_engine import StockRecommendationEngine # Added import


# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

def fetch_news(symbol):
    """Fetch news articles for a given stock symbol"""
    # This is a placeholder - in production, use a real news API
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={st.secrets['ALPHA_VANTAGE_KEY']}"
        response = requests.get(url)
        data = response.json()
        return data.get('feed', [])[:5]  # Get latest 5 news articles
    except:
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def main():
    # Apply custom styles
    apply_custom_styles()

    # Header
    st.title("📈 Advanced Stock Market Predictor")
    st.markdown("""
    This application predicts stock prices using machine learning and provides
    comprehensive technical analysis and portfolio tracking.
    """)

    # Sidebar inputs
    with st.sidebar:
        st.header("Configuration")

        # Dark mode toggle
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        dark_mode = st.toggle('Dark Mode', value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.markdown("""
                <style>
                    :root {
                        --background-color: """ + ('#1E1E1E' if dark_mode else '#FFFFFF') + """;
                        --text-color: """ + ('#FFFFFF' if dark_mode else '#262730') + """;
                        --border-color: """ + ('#363636' if dark_mode else '#E0E0E0') + """;
                    }
                    .stApp {
                        background-color: var(--background-color);
                        color: var(--text-color);
                    }
                </style>
            """, unsafe_allow_html=True)

        # Multiple stock selection
        symbols = st.multiselect(
            "Select Stocks",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            default=["AAPL"]
        )

        timeframe = st.selectbox(
            "Select Timeframe",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            index=6
        )

        interval = st.selectbox(
            "Select Interval",
            ["1d", "1h", "15m", "5m"],
            index=0
        )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([ # Updated tabs
        "📊 Price Prediction",
        "📈 Technical Indicators",
        "💼 Portfolio Tracker",
        "📰 News & Sentiment",
        "🤖 AI Recommendations"
    ])

    try:
        # Tab 1: Price Prediction
        with tab1:
            for symbol in symbols:
                st.subheader(f"{symbol} Price Prediction")

                # Fetch and prepare data
                with st.spinner('Fetching stock data...'):
                    df = fetch_stock_data(symbol, period=timeframe, interval=interval)

                # Display current stock info
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                price_change_pct = (price_change / df['Close'].iloc[-2]) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    show_metric_card("Current Price", f"${current_price:.2f}")
                with col2:
                    show_metric_card("Price Change", f"${price_change:.2f}")
                with col3:
                    show_metric_card("Change %", f"{price_change_pct:.2f}%")

                # Prepare data and train model
                X, y, enhanced_df = prepare_data(df)
                predictor = StockPredictor()

                with st.spinner('Training model...'):
                    model_metrics = predictor.train(X, y)

                # Make predictions
                last_data = X.iloc[-1:]
                predictions, conf_intervals = predictor.predict(last_data)

                # Visualization
                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Historical',
                    line=dict(color='#00B4D8' if dark_mode else 'blue')
                ))

                # Predictions
                future_dates = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=7,
                    freq='B'
                )

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=[predictions[0]] * len(future_dates),
                    name='Predicted',
                    line=dict(color='#FF4B4B' if dark_mode else 'red', dash='dash')
                ))

                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=[conf_intervals['upper'][0]] * len(future_dates),
                    fill=None,
                    mode='lines',
                    line=dict(color='gray', width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=[conf_intervals['lower'][0]] * len(future_dates),
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='gray', width=0),
                    name='Confidence Interval'
                ))

                fig.update_layout(
                    title=f'{symbol} Stock Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    template='plotly_dark' if dark_mode else 'plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Model metrics
                metrics = calculate_metrics(
                    model_metrics['y_test'],
                    model_metrics['test_pred']
                )

                st.subheader("Model Performance Metrics")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Mean Absolute Error", f"${metrics['MAE']}")
                with cols[1]:
                    st.metric("Root Mean Squared Error", f"${metrics['RMSE']}")
                with cols[2]:
                    st.metric("R² Score", f"{metrics['R2 Score']}")

        # Tab 2: Technical Indicators
        with tab2:
            for symbol in symbols:
                st.subheader(f"{symbol} Technical Analysis")

                # Fetch and prepare data with technical indicators
                with st.spinner('Calculating technical indicators...'):
                    df = fetch_stock_data(symbol, period=timeframe, interval=interval)
                    _, _, enhanced_df = prepare_data(df)

                # Technical indicators selection
                indicators = st.multiselect(
                    "Select Technical Indicators",
                    ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages', 'Stochastic'],
                    default=['RSI', 'MACD']
                )

                for indicator in indicators:
                    if indicator == 'RSI':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['RSI'],
                            name='RSI'
                        ))
                        fig.add_hline(y=70, line_dash="dash", line_color="red")
                        fig.add_hline(y=30, line_dash="dash", line_color="green")
                        fig.update_layout(title='Relative Strength Index (RSI)')
                        st.plotly_chart(fig, use_container_width=True)

                    elif indicator == 'MACD':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['MACD'],
                            name='MACD'
                        ))
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['MACD_Signal'],
                            name='Signal Line'
                        ))
                        fig.add_trace(go.Bar(
                            x=enhanced_df.index,
                            y=enhanced_df['MACD_Hist'],
                            name='Histogram'
                        ))
                        fig.update_layout(title='MACD')
                        st.plotly_chart(fig, use_container_width=True)

                    elif indicator == 'Bollinger Bands':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['BB_Upper'],
                            name='Upper Band'
                        ))
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['BB_Middle'],
                            name='Middle Band'
                        ))
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['BB_Lower'],
                            name='Lower Band'
                        ))
                        fig.add_trace(go.Scatter(
                            x=enhanced_df.index,
                            y=enhanced_df['Close'],
                            name='Close Price'
                        ))
                        fig.update_layout(title='Bollinger Bands')
                        st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Portfolio Tracker
        with tab3:
            st.subheader("Portfolio Tracker")

            # Initialize portfolio in session state
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = {}

            # Add new position
            with st.form("add_position"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_symbol = st.text_input("Stock Symbol")
                with col2:
                    quantity = st.number_input("Quantity", min_value=0.0)
                with col3:
                    entry_price = st.number_input("Entry Price", min_value=0.0)

                if st.form_submit_button("Add to Portfolio"):
                    if new_symbol and quantity and entry_price:
                        st.session_state.portfolio[new_symbol] = {
                            'quantity': quantity,
                            'entry_price': entry_price
                        }

            # Display portfolio
            if st.session_state.portfolio:
                portfolio_data = []
                for symbol, data in st.session_state.portfolio.items():
                    try:
                        current_data = fetch_stock_data(symbol, period='1d')
                        current_price = current_data['Close'].iloc[-1]
                        position_value = data['quantity'] * current_price
                        cost_basis = data['quantity'] * data['entry_price']
                        profit_loss = position_value - cost_basis
                        profit_loss_pct = (profit_loss / cost_basis) * 100

                        portfolio_data.append({
                            'Symbol': symbol,
                            'Quantity': data['quantity'],
                            'Entry Price': f"${data['entry_price']:.2f}",
                            'Current Price': f"${current_price:.2f}",
                            'Position Value': f"${position_value:.2f}",
                            'Profit/Loss': f"${profit_loss:.2f}",
                            'Profit/Loss %': f"{profit_loss_pct:.2f}%"
                        })

                    except Exception as e:
                        st.error(f"Error fetching data for {symbol}: {str(e)}")

                if portfolio_data:
                    st.dataframe(pd.DataFrame(portfolio_data))

        # Tab 4: News & Sentiment
        with tab4:
            for symbol in symbols:
                st.subheader(f"{symbol} News & Sentiment Analysis")

                news_articles = fetch_news(symbol)
                if news_articles:
                    sentiments = []
                    for article in news_articles:
                        sentiment = analyze_sentiment(article.get('title', '') + ' ' + article.get('summary', ''))
                        sentiments.append(sentiment)

                        with st.expander(article.get('title', 'No title')):
                            st.write(article.get('summary', 'No summary available'))
                            st.caption(f"Source: {article.get('source', 'Unknown')}")
                            st.caption(f"Sentiment Score: {sentiment:.2f}")

                    # Plot sentiment distribution
                    fig = px.histogram(
                        x=sentiments,
                        nbins=20,
                        title=f"Sentiment Distribution for {symbol} News"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No recent news articles found.")

        recommendation_engine = StockRecommendationEngine() #Added recommendation engine initialization

        # Tab 5: AI Recommendations
        with tab5:
            st.subheader("AI-Powered Stock Recommendations")

            for symbol in symbols:
                st.write(f"### Analysis for {symbol}")

                with st.spinner(f'Analyzing {symbol}...'):
                    # Get news articles for sentiment analysis
                    news_articles = fetch_news(symbol)

                    # Get recommendation
                    recommendation = recommendation_engine.get_recommendation(symbol, news_articles)

                    # Display recommendation
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Recommendation", recommendation['recommendation'])
                    with col2:
                        st.metric("Strength", recommendation['strength'])
                    with col3:
                        st.metric("Score", f"{recommendation['score']:.2f}")

                    # Display detailed analysis
                    with st.expander("See detailed analysis"):
                        # Technical Analysis
                        st.write("#### Technical Analysis")
                        for signal in recommendation['analysis']['technical']:
                            icon = "🟢" if signal['signal'] > 0 else "🔴"
                            st.write(f"{icon} {signal['indicator']}: {signal['reason']}")

                        # Trend Analysis
                        st.write("#### Trend Analysis")
                        for reason in recommendation['analysis']['trend']:
                            st.write(f"• {reason}")

                        # Sentiment Analysis
                        st.write("#### Sentiment Analysis")
                        for reason in recommendation['analysis']['sentiment']:
                            st.write(f"• {reason}")


    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your inputs and try again.")

    # Add footer
    show_footer()

if __name__ == "__main__":
    main()