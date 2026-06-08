from anthropic import Anthropic


class AnthropicProvider:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate_stream_response(self, history: list):
        system_prompt = ""
        clean_messages = []
        for msg in history:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                clean_messages.append(msg)

        response = self.client.messages.create(
            max_tokens=1024,
            model="claude-sonnet-4-0",
            system=system_prompt,
            messages=clean_messages,
            stream=True,
        )

        for event in response:
            if event.type == "content_block_delta":
                text = event.delta.text
                if text:
                    yield text
