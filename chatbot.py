import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"


def answer_college_query(query, index, chunks):
    from rag import retrieve

    context = "\n".join(retrieve(query, index, chunks))

    prompt = f"""You are a helpful college assistant.
Answer using ONLY the context below.

If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def answer_study_query(query):

    prompt = f"""You are a friendly, patient study tutor helping a college student.

Explain clearly and simply with examples.

Student's question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
