# Dubai-Property-AI

## 🏗️ Dubai Property Intelligence (DPI)
AI-Powered Real Estate Investment Audit Tool
Live Demo: https://huggingface.co/spaces/dula22438/Dubai-property-advisor

Developer: Dulasi Nethma

🌟 Overview
Dubai Property Intelligence is a specialized RAG (Retrieval-Augmented Generation) pipeline designed to provide "Decision-Grade" intelligence for the Dubai luxury real estate market.

Unlike standard property listings, this system cross-references active listings (like those from Emaar's 'The Oasis') against raw Dubai Land Department (DLD) transaction data to provide an unbiased, data-backed verdict on property value.

🚀 Key Features
DLD Validation: Automatically cross-references property specifications against actual historical sales records.

Investment Insights: Utilizes Llama 3.1 via Groq API to calculate market velocity and price-per-square-foot benchmarks.

Growth Analysis: Identifies localized trends, such as the current 0.22% market stabilization and scarcity premiums for waterfront assets.

Conversational Interface: A user-friendly Gradio UI for agents and investors to query complex datasets.

🛠️ Tech Stack
Language: Python

LLM: Llama 3.1 (70B Versatile)

API: Groq Cloud

Interface: Gradio

Data Handling: Pandas (for DLD CSV analysis)

Deployment: Hugging Face Spaces

⚙️ Setup & Security
This project implements industry-standard security practices:

Environment Variables: All API keys are managed through secure Secrets/Environment Variables.

Template System: A .env.example file is provided for local configuration.

Safety: Private credentials are excluded from the repository via .gitignore.

Developed as a prototype for advanced PropTech solutions in the UAE market.
