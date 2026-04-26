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
    if not user_message:
        return "", history
    
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
        response = chat_completion.choices[0].message.content
        history.append((user_message, response))
        return "", history 

    except Exception as e:
        history.append((user_message, f"System Error: {str(e)}"))
        return "", history

# Helper function for custom example buttons
def load_example(example_text):
    return example_text

with open("style.css", "r") as f:
    custom_css = f.read()

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🇦🇪 Dubai Property Intelligence")
    gr.Markdown("### Premium AI Insights for Investors & Market Enthusiasts")
    
    with gr.Group(elem_id="chatbot-container"):
        chatbot = gr.Chatbot(elem_id="chatbot", show_label=False)
        msg = gr.Textbox(
            placeholder="Ask me anything about Dubai real estate...",
            show_label=False,
            container=False,
            elem_id="msg-box"
        )
        
        # CUSTOM EXAMPLES (Replacing gr.Examples to stop the crash)
        with gr.Row(elem_id="custom-examples"):
            ex1 = gr.Button("Marina Trends", variant="secondary", size="sm")
            ex2 = gr.Button("Highest ROI", variant="secondary", size="sm")
            ex3 = gr.Button("2026 Projects", variant="secondary", size="sm")
            ex4 = gr.Button("Golden Visa", variant="secondary", size="sm")

    # Mapping buttons to the input box
    ex1.click(lambda: "What are the current property price trends in Dubai Marina?", None, msg)
    ex2.click(lambda: "Which areas in Dubai offer the highest rental yields (ROI) right now?", None, msg)
    ex3.click(lambda: "Tell me about the new off-plan projects launching in 2026.", None, msg)
    ex4.click(lambda: "What are the golden visa requirements for property investors?", None, msg)

    msg.submit(dubai_ai_advisor, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(show_api=False)