import gradio as gr
from groq import Groq
import os  

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load market data
try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except:
    market_context = "Dubai Real Estate data is currently being updated."

def chat_function(message, history):
    try:
        # Build the message list for the LLM
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Use this data: {market_context}"}]
        
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        messages.append({"role": "user", "content": message})

        # Call Llama 3
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# Create the Interface using the most stable component in Gradio
demo = gr.ChatInterface(
    fn=chat_function,
    title="Dubai Property Intelligence",
    description="Ask about market trends, ROI, and off-plan projects.",
    theme="glass" # Built-in theme, no CSS file needed
)

if __name__ == "__main__":
    # show_api=False is crucial to avoid the error you saw
    demo.launch(show_api=False)