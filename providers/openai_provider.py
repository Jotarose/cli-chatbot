from openai import OpenAI, OpenAIError

from core.chatbot import ProviderError


class OpenAIProvider:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_stream_response(self, history: list):
        try:
            response = self.client.responses.create(
                model="gpt-4o",
                input=history,
                stream=True,
            )

            for event in response:
                if event.type == "response.output_text.delta":
                    text = event.delta.text
                    if text:
                        yield text
        except OpenAIError as e:
            raise ProviderError(f"OpenAI API error: {e}")
