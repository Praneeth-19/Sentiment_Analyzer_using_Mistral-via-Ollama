import streamlit as st
import requests

st.title("Sentiment Analyzer (Mistral)")
text_input = st.text_area("Enter your sentence here:")

if st.button("Analyze"):
    try:
        res = requests.post("http://localhost:8000/analyze/", data={"text": text_input})
        
        if res.status_code == 200:
            try:
                sentiment = res.json().get("sentiment", "Error")
                st.subheader("Predicted Sentiment:")
                st.write(sentiment)
            except requests.exceptions.JSONDecodeError:
                st.error(f"Error: Received invalid response from backend. Response text: {res.text[:100]}...")
        elif res.status_code == 504:
            st.error("The model is taking too long to respond. This might happen when the model is loading for the first time. Please try again with a shorter text or wait a moment and try again.")
        else:
            st.error(f"Error: Backend returned status code {res.status_code}. Response: {res.text[:100]}...")
            
    except requests.exceptions.ConnectionError:
        st.error("Error: Cannot connect to the backend server. Please make sure the backend server is running on port 8001.")
