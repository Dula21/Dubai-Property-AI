import gradio as gr
from groq import Groq
import os  



api_key = os.getenv("GROQ_API_KEY") 

client = Groq(api_key=api_key)

def dubai_ai_advisor(user_message, history):
    try:
        # Load the data summary
        with open("market_intelligence.txt", "r") as f:
            market_context = f.read()
        
        system_prompt = f"You are a Dubai Real Estate expert. Use this data: {market_context}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# 2. Interface
demo = gr.ChatInterface(fn=dubai_ai_advisor, title="Dubai Property Intelligence")

# 3. Launch (Note: No share=True needed here!)
if __name__ == "__main__":
    demo.launch()