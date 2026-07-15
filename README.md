# Google Gemini AI Chatbot

A Streamlit web app that lets you chat with Google's Gemini models directly using your own API key. Choose between `gemini-flash-latest`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, and `gemini-2.5-pro`, customize the assistant's persona with a system prompt, and save/restore your conversation history — all with automatic fallback handling if a model is temporarily unavailable.

## Features
- 🔑 Bring your own Google AI Studio API key (never stored or committed)
- 🧠 Switch between four Gemini models on the fly
- 🎭 Set a custom persona/system prompt
- 💾 Download and restore chat history as JSON
- 🔁 Automatic fallback to `gemini-flash-latest` if your selected model fails
- 🔍 Built-in debug log showing exactly which model handled each request

## Setup
\```bash
pip install -r requirements.txt
streamlit run gemini_chatbot.py
\```

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
