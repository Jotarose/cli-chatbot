from google import genai


class GeminiProvider:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate_stream_response(self, history: list):
        response = self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=history,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
