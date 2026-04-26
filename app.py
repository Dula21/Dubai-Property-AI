import gradio as gr
from groq import Groq
import os  

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    with open("market_intelligence.txt", "r") as f:
        market_context = f.read()
except Exception:
    market_context = "Data currently unavailable."

def dubai_ai_advisor(user_message, history):
    try:
        messages = [{"role": "system", "content": f"You are a Dubai Real Estate expert. Use this data: {market_context}"}]
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

with gr.Blocks(css="style.css", theme=gr.themes.Default(primary_hue="amber")) as demo:
    gr.Markdown("# 🇦🇪 Dubai Property Intelligence")
    gr.Markdown("### Premium AI Insights for Investors & Market Enthusiasts")
    
    with gr.Group(elem_id="chatbot-container"):
        chat = gr.ChatInterface(
            fn=dubai_ai_advisor,
            # This ensures the box clears after the message is sent
            autofocus=True,
            examples=[
                ["What are the current property price trends in Dubai Marina?"],
                ["Which areas in Dubai offer the highest rental yields (ROI) right now?"],
                ["Tell me about the new off-plan projects launching in 2026."],
                ["What are the golden visa requirements for property investors?"]
            ]
        )

if __name__ == "__main__":
    demo.launch()