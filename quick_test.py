from scripts.inference import RetailAssistant

def quick_test():
    assistant = RetailAssistant()

    test_cases = [
        "What's the price of Coach sunglasses?",
        "Do you have any red dresses?",
        "What scarfs do you have in gold?",
    ]

    for query in test_cases:
        print(f"\nQuery: {query}")
        response = assistant.generate_response(
            "You are a helpful Macy's virtual shopping assistant. Help the customer find products and provide detailed information.",
            query,
            max_length=100
        )
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    quick_test()
