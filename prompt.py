import openai
import tempfile
import os

class OpenAIConfig:
    def __init__(self, api_key: str = "api", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
        self.conversation_history = [{"role": "system", "content": """You are a professional AI assistant specialized in Swedish insurance only.

Rules:
- Answer strictly according to Swedish insurance systems.
- If information is missing, ask a follow-up question instead of guessing.
- Keep response in English unless the user asks in Swedish.
            """}]

    def get_response(self, prompt: str, history: list) -> str:
        response = openai.chat.completions.create(
            model=self.model,
            messages=history + [{"role": "user", "content": prompt}],
            max_completion_tokens=500
        )

        reply = response.choices[0].message.content
        print("AI Response:", reply)
        return reply

    

    def get_history(self):
        return self.conversation_history