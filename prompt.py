import openai
import tempfile
import os

class OpenAIConfig:
    def __init__(self, api_key: str = "api", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
        self.conversation_history = [{"role": "system", "content": """You are a warm, friendly insurance advisor. 
            Respond naturally and conversationally.
            Ask only one focused question OR give one clear next step per reply. Never ask multiple questions or chain follow-ups in the same response. 
            If the user reports an incident then ask one follow-up question that is necessary to proceed (e.g., "Can you tell me a bit more about what happened?"). Stop after that single question.
            Keep replies short by default unless the user explicitly asks for more detail or explanation.
            If a user message is NOT related to insurance or insurance assistance, politely decline and guide them back to an insurance-related topic.
            """}]

    def get_response(self, prompt: str, history: list) -> str:
        response = openai.chat.completions.create(
        model=self.model,
        messages=history + [{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5,
    )

        reply = response.choices[0].message.content
        return reply

    

    def get_history(self):
        return self.conversation_history