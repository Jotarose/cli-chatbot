import os

from dotenv import load_dotenv

from core import ConversationManager, FallbackChatbot
from providers import AnthropicProvider, GeminiProvider, OpenAIProvider

load_dotenv()


def main():
    conversation_manager = ConversationManager()
    providers = [
        OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
        AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY")),
        GeminiProvider(api_key=os.getenv("GEMINI_API_KEY")),
    ]
    conversation_manager.add_message(
        "system",
        "You are a helpful assistant that provides concise and accurate answers to user queries. If you don't know the answer, say you don't know instead of making something up. All answers should be in Spanish.",
    )
    chatbot = FallbackChatbot(
        providers=providers, conversation_manager=conversation_manager
    )

    print("Hello from cli-chatbot!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["/salir"]:
            print("Adios!")
            break

        print("Chatbot:", end=" ", flush=True)
        for chunk in chatbot.get_response(user_input):
            print(chunk, end="", flush=True)
        print()  # for newline after response


if __name__ == "__main__":
    main()
