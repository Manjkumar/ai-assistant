import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mac_config import mac_config

class RetailAssistantQuantized:
    def __init__(self, model_path="models/retail-assistant"):
        self.device = torch.device("cpu")  # Use CPU for quantization
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model with quantization"""
        print(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load base model on CPU
        base_model_name = "microsoft/phi-2"
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Load LoRA adapter
        print(f"Loading LoRA adapter from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)

        # Move to CPU and eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply dynamic quantization (Mac-compatible)
        print("Applying dynamic quantization...")
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize all Linear layers
            dtype=torch.qint8    # Use 8-bit integers
        )

        print("Quantized model loaded successfully!")

    def generate_response(self, instruction, user_input, max_length=100):
        """Generate response from the quantized model"""
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
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )

        # Decode response
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part after ### Response:
        if "### Response:" in full_text:
            parts = full_text.split("### Response:", 1)
            if len(parts) > 1:
                response_part = parts[1]
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
        print("\n🛍️ Welcome to Macy's Virtual Assistant (Quantized)!")
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
    assistant = RetailAssistantQuantized()
    assistant.chat()

if __name__ == "__main__":
    main()
