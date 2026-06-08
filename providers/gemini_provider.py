from google import genai
from google.genai import types


class GeminiProvider:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate_stream_response(self, history: list):
        system_prompt = ""
        gemini_history = []
        for msg in history:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"

            elif msg["role"] == "assistant":
                gemini_history.append(
                    {
                        "role": "model",
                        "parts": [{"text": msg["content"]}],
                    }
                )
            else:
                gemini_history.append(
                    {
                        "role": msg["role"],
                        "parts": [{"text": msg["content"]}],
                    }
                )

        response = self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=gemini_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
