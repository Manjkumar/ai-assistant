import json
import jsonlines
import random
from datetime import datetime, timedelta

# Sample retail data generator
class RetailDataGenerator:
    def __init__(self):
        self.products = {
            "clothing": {
                "items": ["dress", "shirt", "jeans", "jacket", "sweater", "blazer", "coat"],
                "brands": ["Calvin Klein", "Ralph Lauren", "Tommy Hilfiger", "Levi's", "Nike"],
                "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
                "colors": ["black", "white", "blue", "red", "green", "navy", "gray"]
            },
            "shoes": {
                "items": ["sneakers", "boots", "heels", "flats", "sandals", "loafers"],
                "brands": ["Nike", "Adidas", "Steve Madden", "Cole Haan", "UGG"],
                "sizes": ["6", "7", "8", "9", "10", "11", "12"],
                "colors": ["black", "white", "brown", "tan", "navy"]
            },
            "accessories": {
                "items": ["handbag", "wallet", "belt", "watch", "sunglasses", "scarf"],
                "brands": ["Coach", "Michael Kors", "Kate Spade", "Fossil", "Ray-Ban"],
                "colors": ["black", "brown", "tan", "gold", "silver"]
            }
        }

        self.stores = [
            {"name": "Herald Square", "address": "151 W 34th St, New York, NY", "hours": "10AM-9PM"},
            {"name": "Brooklyn Downtown", "address": "422 Fulton St, Brooklyn, NY", "hours": "10AM-8PM"},
            {"name": "Roosevelt Field", "address": "630 Old Country Rd, Garden City, NY", "hours": "10AM-9:30PM"}
        ]

    def generate_training_samples(self, num_samples=1000):
        samples = []

        # Product inquiry samples
        for _ in range(num_samples // 4):
            category = random.choice(list(self.products.keys()))
            item = random.choice(self.products[category]["items"])
            brand = random.choice(self.products[category]["brands"])
            color = random.choice(self.products[category]["colors"])

            if category == "shoes":
                size = random.choice(self.products[category]["sizes"])
                size_text = f"size {size}"
            elif category == "clothing":
                size = random.choice(self.products[category]["sizes"])
                size_text = f"size {size}"
            else:
                size_text = ""

            price = random.randint(20, 500)

            # Create conversational Q&A
            questions = [
                f"Do you have {color} {item} from {brand}?",
                f"What {item}s do you have in {color}?",
                f"Show me {brand} {item}s",
                f"I'm looking for a {color} {item} {size_text}",
                f"What's the price of {brand} {item}?"
            ]

            responses = [
                f"Yes, we have {brand} {item} in {color} {size_text}. It's currently priced at ${price}. Would you like to see more details or check availability in store?",
                f"We have several {color} {item}s available. The {brand} option is ${price} and {size_text if size_text else 'available in multiple sizes'}. Can I help you with sizing or styling advice?",
                f"Great choice! Our {brand} {item} collection includes {color} options starting at ${price}. {size_text.capitalize() if size_text else 'Available in various sizes'}. Would you like to check store availability?"
            ]

            samples.append({
                "instruction": "You are a helpful Macy's virtual shopping assistant. Help the customer find products and provide detailed information.",
                "input": random.choice(questions),
                "output": random.choice(responses)
            })

        # Order tracking samples
        for i in range(num_samples // 4):
            order_num = f"MCS{random.randint(1000000, 9999999)}"
            days_ago = random.randint(1, 7)
            delivery_date = (datetime.now() + timedelta(days=random.randint(2, 5))).strftime("%B %d")

            tracking_questions = [
                f"Where is my order {order_num}?",
                "Can you track my order?",
                f"What's the status of order {order_num}?",
                "When will my order arrive?"
            ]

            tracking_responses = [
                f"Your order {order_num} was shipped {days_ago} days ago and is currently in transit. Expected delivery is {delivery_date}. You can track it using the link sent to your email.",
                f"I can see that order {order_num} is on its way! It should arrive by {delivery_date}. The package is currently with our carrier partner.",
                f"Order {order_num} is being processed and will be delivered by {delivery_date}. You'll receive a notification when it's out for delivery."
            ]

            samples.append({
                "instruction": "You are a Macy's customer service assistant helping with order tracking.",
                "input": random.choice(tracking_questions),
                "output": random.choice(tracking_responses)
            })

        # Store information samples
        for _ in range(num_samples // 4):
            store = random.choice(self.stores)

            store_questions = [
                f"What are the hours for the {store['name']} store?",
                f"Where is your {store['name']} location?",
                "What time does Macy's close today?",
                "Find me the nearest Macy's store"
            ]

            store_responses = [
                f"The {store['name']} Macy's is located at {store['address']} and is open {store['hours']} today. Would you like directions?",
                f"Our {store['name']} store is open {store['hours']}. The address is {store['address']}. Can I help you check product availability there?",
                f"The closest Macy's to you is at {store['address']} ({store['name']}), open {store['hours']}. Would you like to call ahead for product availability?"
            ]

            samples.append({
                "instruction": "You are a Macy's assistant providing store location and hours information.",
                "input": random.choice(store_questions),
                "output": random.choice(store_responses)
            })

        # Sales and promotions samples
        for _ in range(num_samples // 4):
            discount = random.choice([20, 25, 30, 40, 50])
            category = random.choice(list(self.products.keys()))

            promo_questions = [
                "What sales are happening now?",
                f"Any discounts on {category}?",
                "Do you have any coupons?",
                "What's on clearance?"
            ]

            promo_responses = [
                f"Great timing! We have {discount}% off on all {category} this week. Plus, use code SAVE10 for an extra 10% off. Would you like to see specific items?",
                f"Yes! Our {category} department has items up to {discount}% off. Star Rewards members get an additional 5% off. Can I help you find something specific?",
                f"We're running our seasonal sale with {discount}% off select {category}. Free shipping on orders over $49. What are you shopping for today?"
            ]

            samples.append({
                "instruction": "You are a Macy's assistant helping customers find deals and promotions.",
                "input": random.choice(promo_questions),
                "output": random.choice(promo_responses)
            })

        return samples

# Generate dataset
generator = RetailDataGenerator()
training_data = generator.generate_training_samples(1000)

# Save as JSONL
with jsonlines.open('data/retail_training.jsonl', 'w') as writer:
    writer.write_all(training_data)

# Also save a validation set
validation_data = generator.generate_training_samples(200)
with jsonlines.open('data/retail_validation.jsonl', 'w') as writer:
    writer.write_all(validation_data)

print(f"Generated {len(training_data)} training samples")
print(f"Generated {len(validation_data)} validation samples")
print("\nSample data:")
print(json.dumps(training_data[0], indent=2))