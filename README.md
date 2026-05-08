# How to start the project

Steps:

- clone project
- activate virtual environment on windows in the terminal (venv\Scripts\activate)
- install all dependencies from requirements.txt
- add .env file with LLM settings:
  - LLM_PROVIDER=mistral
  - LLM_MODEL=mistral-large-latest
  - MISTRAL_API_KEY=your-api-key
- run project in the terminal

```bash
uv run streamlit run app.py 
```

- to exit virtual environment in the terminal execute this command 

```bash
deactivate
```

To use Gemini instead, set:

- LLM_PROVIDER=gemini
- LLM_MODEL=gemini-2.5-flash
- GEMINI_API_KEY=your-api-key

To use OpenRouter with DeepSeek instead, set:

- LLM_PROVIDER=openrouter
- LLM_MODEL=deepseek/deepseek-v4-pro
- OPENROUTER_API_KEY=your-api-key
