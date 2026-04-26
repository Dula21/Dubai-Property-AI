import gradio as gr
from groq import Groq
import os  

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except:
    market_context = "Dubai Real Estate market data."

def chat_function(message, history):
    try:
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Context: {market_context}"}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Clean Interface without the "clogging" examples
demo = gr.ChatInterface(
    fn=chat_function,
    title="🇦🇪 Dubai Property Intelligence",
    description="Your premium advisor for ROI, market trends, and off-plan investments.",
    theme="soft",
    css="style.css",
    # We remove the examples= parameter here to keep the input box empty
)

if __name__ == "__main__":
    demo.launch(show_api=False)