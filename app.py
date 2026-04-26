import gradio as gr
from groq import Groq
import os  

# 1. Setup Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load market data once at startup
try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except Exception:
    market_context = "Data currently unavailable."

# 2. Custom CSS for Fonts and Styling
# We are using 'Montserrat' for a premium Dubai real estate feel
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

footer {visibility: hidden}
.gradio-container {
    font-family: 'Montserrat', sans-serif !important;
}
#component-0 {
    max-width: 900px;
    margin: auto;
}
.primary-header {
    text-align: center;
    color: #C5A059; /* Gold accent */
}
"""

def dubai_ai_advisor(user_message, history):
    try:
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Use this data: {market_context}"}]
        
        # History handling for Gradio 4.36.1 (list of lists)
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
                
        messages.append({"role": "user", "content": user_message})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# 3. Build Interface
with gr.Blocks(css=custom_css, title="DPI: Dubai Property Intelligence") as demo:
    gr.Markdown("# 🇦🇪 Dubai Property Intelligence", elem_classes="primary-header")
    gr.Markdown("### Your specialized AI advisor for ROI, trends, and investments.")
    
    gr.ChatInterface(
        fn=dubai_ai_advisor,
        examples=[
            ["What are the current property price trends in Dubai Marina?"],
            ["Which areas in Dubai offer the highest rental yields (ROI) right now?"],
            ["Tell me about the new off-plan projects launching in 2026."],
            ["What are the golden visa requirements for property investors?"]
        ]
    )

# 4. Launch
if __name__ == "__main__":
    demo.launch()