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
        # 1. Start with the expert persona
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Use this data: {market_context}"}]
        
        # 2. Add history (In Gradio 4.36.1, history is a list of lists)
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
                
        # 3. Add the current message
        messages.append({"role": "user", "content": user_message})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"
    
    
# 2. Interface
demo = gr.ChatInterface(
    fn=dubai_ai_advisor, 
    
    title="DPI: Dubai Property Intelligence 🇦🇪",
    description="I am an AI trained on Dubai Land Department trends. Ask me anything about property prices, ROI, or market trends!",
    
    cache_examples=False,
    theme=gr.themes.Soft()  # Fix: use object not string
)

# 3. Launch
if __name__ == "__main__":
      demo.launch()