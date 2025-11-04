import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mac_config import mac_config

class RetailAssistant:
    def __init__(self, model_path="models/retail-assistant"):
        self.device = mac_config.get_device()
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load base model
        base_model_name = "microsoft/phi-2"
        print(f"Loading base model: {base_model_name}")
        with mac_config.mutex_guard():
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )

        # Load LoRA adapter
        print(f"Loading LoRA adapter from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.to(self.device)

        self.model.eval()

        print("Model loaded successfully!")

    def generate_response(self, instruction, user_input, max_length=100):
        """Generate response from the model"""
        # Format prompt
        prompt = f"""### Instruction: {instruction}
### Input: {user_input}
### Response:"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate with optimizations for speed
        # Optimizations: greedy decoding, KV cache, reduced max tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,  # Greedy decoding is faster than sampling
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1  # No beam search for speed
            )

        # Decode response
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part after ### Response:
        if "### Response:" in full_text:
            # Split on ### Response: and get everything after the first occurrence
            parts = full_text.split("### Response:", 1)
            if len(parts) > 1:
                response_part = parts[1]
                # Now stop at the next ### marker (which would be a new instruction/input)
                if "\n###" in response_part:
                    response = response_part.split("\n###")[0].strip()
                else:
                    response = response_part.strip()
            else:
                response = full_text
        else:
            response = full_text

        return response

    def chat(self):
        """Interactive chat interface"""
        print("\n🛍️ Welcome to Macy's Virtual Assistant!")
        print("I can help you with:")
        print("- Finding products and checking availability")
        print("- Tracking orders")
        print("- Store locations and hours")
        print("- Current sales and promotions")
        print("\nType 'quit' to exit\n")

        instruction = "You are a helpful Macy's virtual shopping assistant. Provide friendly, accurate, and helpful responses to customer inquiries."

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for shopping with Macy's! Have a wonderful day! 👋")
                break

            if not user_input:
                continue

            response = self.generate_response(instruction, user_input)
            print(f"\nAssistant: {response}")

def main():
    assistant = RetailAssistant()
    assistant.chat()

if __name__ == "__main__":
    main()