import gradio as gr
from groq import Groq
import os  

# 1. Setup Client
api_key = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=api_key)

# Load market data once at startup (not on every request)
try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except FileNotFoundError:
    market_context = "No market data available."

def dubai_ai_advisor(user_message, history):
    try:
        system_prompt = f"""You are a Dubai Real Estate expert AI assistant. 
        Use the following market data to answer questions accurately:
        
        {market_context}
        
        Always provide specific data points when available and be concise."""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# 2. Interface
demo = gr.ChatInterface(
    fn=dubai_ai_advisor, 
    type="messages",
    title="DPI: Dubai Property Intelligence 🇦🇪",
    description="I am an AI trained on Dubai Land Department trends. Ask me anything about property prices, ROI, or market trends!",
    examples=[
        ["What are the latest trends in Downtown?"], 
        ["Is it a good time to buy in JVC?"],
        ["Which area has the best ROI for apartments?"],
        ["Tell me about villa prices in Dubai Hills."]
    ],
    cache_examples=False,
    theme=gr.themes.Soft()  # Fix: use object not string
)

# 3. Launch
if __name__ == "__main__":
      demo.launch()