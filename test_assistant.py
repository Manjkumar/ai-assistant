from scripts.inference import RetailAssistant

def test_assistant():
    assistant = RetailAssistant()

    test_cases = [
        "Do you have any red dresses?",
        "What are the store hours for Herald Square?",
        "Track my order MCS1234567",
        "What's on sale today?",
        "I need a birthday gift for my wife",
        "Show me designer handbags under $500"
    ]

    for query in test_cases:
        print(f"\nQuery: {query}")
        response = assistant.generate_response(
            "You are a helpful Macy's assistant",
            query
        )
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_assistant()