# Trading Analyzer (listo para GitHub)

Aplicación web que analiza tickers con regresión, LSTM (ejemplo) y sentimiento (tweets/noticias).

Instalación local:
1. Crear virtualenv: python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. Copiar .env.example a .env y editar con tus claves
4. uvicorn backend.app:app --reload --port 8000

Advertencia: NO subir .env con claves a GitHub.
