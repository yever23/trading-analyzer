# -*- coding: utf-8 -*-
import os
import io
import datetime as dt
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import requests
import base64

# Opcional: para LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

load_dotenv()

TWITTER_BEARER = os.getenv('TWITTER_BEARER_TOKEN')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

# Inicializa el pipeline de sentimiento (Hugging Face)
sentiment_analyzer = pipeline('sentiment-analysis')

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    """Página principal"""
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/api/analyze')
async def analyze(ticker: str = Form(...), prompt_choice: int = Form(...)):
    """Endpoint que realiza el análisis para un ticker y devuelve JSON con resultados y gráfico en base64."""
    ticker = ticker.upper().strip()

    # 1) Obtener precios históricos
    try:
        hist = yf.download(ticker, period='90d', interval='1d', progress=False)
    except Exception as e:
        return JSONResponse({'error': 'Error al obtener historial', 'detail': str(e)}, status_code=400)
    if hist.empty:
        return JSONResponse({'error': 'No se encontró información para el ticker'}, status_code=404)

    current_price = float(hist['Close'][-1])

    # 2) Modelo de regresión simple
    X = np.arange(len(hist)).reshape(-1,1)
    y = hist['Close'].values
    lr = LinearRegression()
    lr.fit(X, y)
    future_days = 14
    X_future = np.arange(len(hist), len(hist)+future_days).reshape(-1,1)
    pred_lr = lr.predict(X_future)
    lr_pred_price_7 = float(pred_lr[min(6, len(pred_lr)-1)])

    # 3) LSTM (ejemplo rápido)
    def lstm_predict(series, days=7, epochs=5):
        arr = np.array(series).astype('float32')
        scaler = StandardScaler()
        arr_s = scaler.fit_transform(arr.reshape(-1,1))
        lookback = 7
        Xs, ys = [], []
        for i in range(len(arr_s)-lookback):
            Xs.append(arr_s[i:i+lookback])
            ys.append(arr_s[i+lookback])
        Xs, ys = np.array(Xs), np.array(ys)
        if len(Xs) < 10:
            return None
        model = Sequential()
        model.add(LSTM(32, input_shape=(Xs.shape[1], Xs.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=Adam(1e-3), loss='mse')
        model.fit(Xs, ys, epochs=epochs, batch_size=8, verbose=0)
        last_seq = arr_s[-lookback:].reshape(1, lookback, 1)
        preds = []
        for _ in range(days):
            p = model.predict(last_seq, verbose=0)[0,0]
            preds.append(p)
            last_seq = np.roll(last_seq, -1)
            last_seq[0,-1,0] = p
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten().tolist()
        return preds

    lstm_pred = lstm_predict(hist['Close'].values, days=14, epochs=3)

    # 4) Tweets (48h) — usa Twitter API v2 si tienes token
    tweets_text = []
    if TWITTER_BEARER:
        url = 'https://api.twitter.com/2/tweets/search/recent'
        headers = {'Authorization': f'Bearer {TWITTER_BEARER}'}
        params = {'query': ticker + ' -is:retweet', 'max_results': '50', 'tweet.fields': 'created_at,text'}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.ok:
                data = r.json()
                for t in data.get('data', []):
                    tweets_text.append(t.get('text',''))
        except Exception:
            pass

    # 5) Noticias (últimas 48h) — NewsAPI si tienes clave
    news_items = []
    if NEWSAPI_KEY:
        nurl = 'https://newsapi.org/v2/everything'
        since = (dt.datetime.utcnow() - dt.timedelta(hours=48)).isoformat()
        params = {'q': ticker, 'from': since, 'sortBy': 'relevancy', 'apiKey': NEWSAPI_KEY, 'pageSize': 20}
        try:
            rn = requests.get(nurl, params=params, timeout=10)
            if rn.ok:
                j = rn.json()
                for art in j.get('articles', []):
                    news_items.append({'title': art.get('title'), 'url': art.get('url'), 'source': art.get('source',{}).get('name')})
        except Exception:
            pass

    # 6) Análisis de sentimiento (muestra)
    combined_texts = tweets_text[:50] + [n['title'] for n in news_items]
    sentiment_summary = {}
    if combined_texts:
        sample = combined_texts[:25]
        try:
            sentiments = sentiment_analyzer(sample)
            pos = sum(1 for s in sentiments if s['label'].lower().startswith('pos'))
            neg = sum(1 for s in sentiments if s['label'].lower().startswith('neg'))
            neu = len(sentiments) - pos - neg
            sentiment_summary = {'positivos': pos, 'negativos': neg, 'neutrales': neu, 'n': len(sentiments)}
        except Exception:
            sentiment_summary = {'error': 'No se pudo procesar sentimiento'}

    # 7) Generar gráfico PNG en base64
    plt.figure(figsize=(10,4))
    plt.plot(hist.index, hist['Close'], label='Close')
    future_index = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=len(pred_lr))
    plt.plot(list(hist.index) + list(future_index), list(hist['Close']) + list(pred_lr), '--', label='Regresión')
    if lstm_pred:
        future_index2 = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=len(lstm_pred))
        plt.plot(future_index2, lstm_pred, ':', label='LSTM')
    plt.legend()
    plt.title(f'{ticker} Precio de cierre y predicciones')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')

    # 8) Señal simple (ejemplo conservador)
    signal = 'HOLD'
    try:
        lr7 = lr_pred_price_7
        lstm7 = float(lstm_pred[6]) if lstm_pred else None
        if lstm7 and lr7 > current_price and lstm7 > current_price:
            signal = 'BUY'
        elif lr7 < current_price and (lstm7 is None or lstm7 < current_price):
            signal = 'SELL'
    except Exception:
        signal = 'HOLD'

    result = {
        'ticker': ticker,
        'precio_actual': current_price,
        'regresion_pred_7d': lr_pred_price_7,
        'lstm_pred_7d': lstm_pred[6] if lstm_pred else None,
        'sentimiento': sentiment_summary,
        'noticias': news_items[:10],
        'tweets_muestra': len(tweets_text),
        'senal': signal,
        'chart_png_b64': img_b64
    }
    return JSONResponse(result)

