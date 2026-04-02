# AI Inventory Risk Agent

An AI-powered supply chain decision support tool that analyzes SKU-level inventory data to identify stockout risk, generate reorder recommendations, and answer planning questions using natural language.

## Features

• Inventory risk classification (Urgent / High / Medium / Low)  
• Reorder quantity recommendations  
• Shortage cost estimation  
• Planner-style AI summaries  
• Natural language inventory queries  
• Interactive Streamlit dashboard

## Tech Stack

Python  
Pandas  
Streamlit  
Groq LLM (Llama 3.3)

## Project Architecture

tools.py
- deterministic supply chain calculations
- inventory metrics and risk scoring

agent.py
- routing logic
- connects user questions to analysis tools

llm_helper.py
- LLM interaction layer
- planner summaries and question answering

app.py
- Streamlit dashboard UI

## Run Locally
