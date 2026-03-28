import os
from groq import Groq


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)


def generate_planner_summary(summary_metrics: dict, context_table: str) -> str:
    client = get_client()

    prompt = f"""
You are a supply chain planning copilot.

Below are portfolio-project inventory analysis results.

Summary metrics:
{summary_metrics}

Top inventory context:
{context_table}

Write a concise planner-style summary with:
1. Biggest inventory risks
2. Most urgent SKUs
3. Likely operational impact
4. Recommended next actions

Keep it practical, professional, and specific.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert supply chain planner and inventory risk analyst.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def answer_inventory_question(user_question: str, summary_metrics: dict, context_table: str) -> str:
    client = get_client()

    prompt = f"""
You are an agentic inventory risk copilot.

You are given computed inventory metrics from a Python analysis engine.
Use those results to answer the user's question accurately.
Do not invent data not present in the context.

Summary metrics:
{summary_metrics}

Inventory context:
{context_table}

User question:
{user_question}

Answer clearly and business-style. If useful, include bullet points.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert supply chain AI copilot.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content