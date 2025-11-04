import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.inference import RetailAssistant

# Initialize assistant
assistant = RetailAssistant()

def respond(message, history):
    """Generate response for Gradio interface"""
    instruction = "You are a helpful Macy's virtual shopping assistant."
    response = assistant.generate_response(instruction, message)
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    respond,
    title="Macy's Virtual Shopping Assistant",
    description="Welcome to Macy's! I'm here to help you find products, track orders, and answer questions about our stores and promotions.",
    examples=[
        "Do you have any black dresses in size M?",
        "What are your store hours?",
        "Where is my order MCS1234567?",
        "What sales are happening now?",
        "Show me Nike sneakers",
        "I need a gift for my mom"
    ],
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="gray"
    )
)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)