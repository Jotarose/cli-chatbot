class FallbackChatbot:
    def __init__(self, providers, conversation_manager):
        self.providers = providers
        self.conversation_manager = conversation_manager

    def get_response(self, user_message: str):
        self.conversation_manager.add_message("user", user_message)
        history = self.conversation_manager.get_history()

        for provider in self.providers:
            try:
                stream = provider.generate_stream_response(history)
                full_response = ""
                for chunk in stream:
                    yield chunk
                    full_response += chunk

                self.conversation_manager.add_message("assistant", full_response)

                return

            except Exception as e:
                print(
                    f"Error occurred while fetching response from {provider.__class__.__name__}: {e}\n- Will try next provider."
                )
        yield "Sorry, I couldn't find a response to your query\nAll providers failed."
