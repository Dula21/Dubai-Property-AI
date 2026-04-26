import gradio as gr
from groq import Groq
import os  

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Context loading remains the same
try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except:
    market_context = "Data unavailable."

def chat_function(message, history):
    try:
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Use: {market_context}"}]
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

# Re-inject the CSS file here
demo = gr.ChatInterface(
    fn=chat_function,
    title="🇦🇪 Dubai Property Intelligence",
    theme="soft",
    css="style.css" # Gradio will look for the separate file here
)

if __name__ == "__main__":
    demo.launch(show_api=False)