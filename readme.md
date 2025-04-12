# AI-Powered Attendance Assistant

This project is an AI-powered chatbot for attendance management that uses a RAG (Retrieval Augmented Generation) pipeline with LangChain to provide natural language understanding and response generation.

## Features

- Natural language understanding for attendance queries
- Seamless integration with existing attendance database
- RAG (Retrieval Augmented Generation) for accurate information retrieval
- Fallback to pattern matching for specific queries
- Modern GUI interface
- Powered by Groq's state-of-the-art LLama3 model

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up your environment variables**

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
```

Then open the `.env` file and add your Groq API key. You can get an API key from [https://console.groq.com/keys](https://console.groq.com/keys).

3. **Run the chatbot**

```bash
python chatbot.py
```

## Architecture

The chatbot uses a tiered approach to answering questions:

1. First attempts to use the RAG pipeline with LangChain and Groq's LLama3 for natural language understanding
2. Falls back to rule-based pattern matching for specific query types
3. Uses a legacy similarity-based model as a final fallback option

The RAG pipeline:
- Converts student, subject, attendance data into vector embeddings
- Uses semantic search to find relevant information
- Leverages Groq's powerful language models to generate natural responses

## Extending the Chatbot

To add new data sources:
1. Update the data loading functions in `train_chatbot.py`
2. Create new DataFrameLoader instances in the `create_vector_stores` method
3. Run the chatbot to refresh the model
