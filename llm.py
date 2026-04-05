import os

import deepseek
from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv()


os.getenv('DEEPSEEK_API_KEY')
os.getenv('DEEPSEEK_API_URL')





class LLM_Client:
    def __init__(self):
        api_key = os.getenv('DEEPSEEK_API_KEY')
        api_url = os.getenv('DEEPSEEK_API_URL')
        MODEL_ID = os.getenv('MODULE_ID')
        self.llm = init_chat_model(
            model=MODEL_ID,
            api_key=api_key,
            base_url=api_url,
        )

