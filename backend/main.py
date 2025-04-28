from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_sentiment(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    logger.info(f"Received text for analysis: {text[:50]}...")
    
    ollama_url = 'http://localhost:11434/api/generate'
    payload = {
        "model": "mistral",
        "prompt": f"Classify the sentiment as Positive, Negative, or Neutral: '{text}'",
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama with payload: {payload}")
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=60)
        logger.info(f"Received response from Ollama with status code: {response.status_code}")
        
        response.raise_for_status()  
        
        try:
            result = response.json()
            logger.info(f"Successfully parsed JSON response: {result}")
            
            sentiment = result.get('response', '').strip()
            if not sentiment:
                logger.warning(f"Ollama returned an empty response field. Full response: {result}")
                raise HTTPException(status_code=500, detail="Ollama returned empty response")
                
            sentiment = sentiment.split()[0]
            logger.info(f"Extracted sentiment: {sentiment}")
            
            return {"sentiment": sentiment}
            
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from Ollama. Status Code: {response.status_code}")
            logger.error(f"Ollama Response Text: {response.text}")
            raise HTTPException(status_code=500, detail="Invalid response format from Ollama")

    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama at {ollama_url}")
        raise HTTPException(status_code=503, detail="Could not connect to Ollama service. Make sure Ollama is running.")
        
    except requests.exceptions.Timeout:
        logger.error("Request to Ollama timed out")
        raise HTTPException(status_code=504, detail="Request to Ollama timed out")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request to Ollama: {e}")
        if 'response' in locals():
            logger.error(f"Ollama Response Status Code: {response.status_code}")
            logger.error(f"Ollama Response Text: {response.text}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")
