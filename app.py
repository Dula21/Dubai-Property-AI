import gradio as gr
from groq import Groq
import os  

# 1. Setup Client
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
demo = gr.ChatInterface(
    fn=dubai_ai_advisor, 
    title="DPI: Dubai Property Intelligence 🇦🇪",
    description="I am an AI trained on Dubai Land Department trends. Ask me anything about property prices, ROI, or market trends!",
    # CRITICAL FIX: Wrapped each example in its own list []
    examples=[
        ["What are the latest trends in Downtown?"], 
        ["Is it a good time to buy in JVC?"],
        ["Which area has the best ROI for apartments?"],
        ["Tell me about villa prices in Dubai Hills."]
    ],
    cache_examples=False,
    theme="soft"
)

# 3. Launch
if __name__ == "__main__":
    # Standard launch for Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)