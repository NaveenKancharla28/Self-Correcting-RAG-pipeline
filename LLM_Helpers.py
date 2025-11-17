from first import LLM_Model, OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def chat(system: str, user: str, model: str = LLM_Model, temperature: float = 0.2) -> str:
    msgs = client.chat.completions.create(
        model_name=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
    )
    return msgs.choices[0].message.content.strip()
    