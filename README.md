# How to start the project

Steps:

- clone project
- activate virtual environment on windows in the terminal (venv\Scripts\activate)
- install all dependencies from requirements.txt
- add .env file with LLM settings:
  - LLM_PROVIDER=mistral
  - LLM_MODEL=mistral-large-latest
  - MISTRAL_API_KEY=your-api-key
- run project with streamlit run app.py in the terminal
- to exit virtual environment in the terminal execute this command deactivate

To use Gemini instead, set:

- LLM_PROVIDER=gemini
- LLM_MODEL=gemini-2.5-flash
- GEMINI_API_KEY=your-api-key
