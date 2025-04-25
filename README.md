# Sentiment Analyzer (Mistral)
A simple AI app that uses the **Mistral model** via Ollama to classify the sentiment
of text.
## Features
- **FastAPI** backend
- **Streamlit** frontend
- **Ollama-hosted Mistral model**
## Run Locally
1. Clone the repo
2. Pull Mistral with Ollama: `ollama pull mistral`
3. Start backend: `uvicorn backend.main:app --reload`
4. Start frontend: `streamlit run frontend/app.py`
