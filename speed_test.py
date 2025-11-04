from scripts.inference import RetailAssistant
import time

def speed_test():
    print("Loading model...")
    start = time.time()
    assistant = RetailAssistant()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s\n")

    test_queries = [
        "What's the price of Coach sunglasses?",
        "Do you have any red dresses?",
        "What scarfs do you have in gold?",
    ]

    times = []
    for query in test_queries:
        print(f"Query: {query}")
        start = time.time()
        response = assistant.generate_response(
            "You are a helpful Macy's virtual shopping assistant. Help the customer find products and provide detailed information.",
            query
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Response ({elapsed:.2f}s): {response}")
        print("-" * 80)

    avg_time = sum(times) / len(times)
    print(f"\nAverage response time: {avg_time:.2f}s")
    print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")

if __name__ == "__main__":
    speed_test()
