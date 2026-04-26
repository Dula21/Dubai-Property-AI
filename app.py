import gradio as gr
from groq import Groq
import os  

# 1. Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Load Market Intelligence Context
try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except Exception:
    market_context = "Data on 2026 Dubai Real Estate trends, ROI, and Golden Visa regulations."

# 3. Define the Chat Logic
def chat_function(message, history):
    try:
        
        system_instructions = (
    f"Current Date: April 2026. You are a senior Dubai Real Estate Advisor. "
    f"Primary Market Intelligence: {market_context}. "
    "When asked about market health, cite the 'Monthly Growth' (0.23%) and 'Average Velocity' (0.70%). "
    "Explain that the market is 'Stabilizing' and advise clients to focus on rental yields "
    "rather than quick flipping. Be professional, data-centric, and authoritative."
)
        
        messages = [{"role": "system", "content": system_instructions}]
        
        # Add conversation history
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # Add the new user message
        messages.append({"role": "user", "content": message})

        # Call the Llama 3 model via Groq
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7, # Balanced for professional yet creative advice
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        return f"⚠️ System Error: {str(e)}"

# 4. Build the Interface
# Using ChatInterface ensures stability against the Gradio 4.36.1 bug
demo = gr.ChatInterface(
    fn=chat_function,
    title="🇦🇪 Dubai Property Intelligence",
    description="Strategic 2026 Real Estate Insights for Global Investors.",
    theme="soft",
    css="style.css", # Link to your external stylesheet
    # Examples are omitted here to keep the input box 'unclogged'
)

# 5. Launch
if __name__ == "__main__":
    # show_api=False prevents the ASGI application crash
    demo.launch(show_api=False)