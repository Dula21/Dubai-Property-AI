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
        # STRATEGIC FIX: We explicitly tell the AI it is 2026 
        # and to use the provided data as the primary source of truth.
        system_instructions = (
            f"Current Date: April 2026. You are a senior Dubai Real Estate Advisor. "
            f"You have been provided with exclusive 2026 market intelligence: {market_context}. "
            "IMPORTANT: Do not use disclaimers about your knowledge cutoff. "
            "Answer confidently using the provided 2026 data. You are a 2026 expert."
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