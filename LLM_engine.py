import os
from groq import Groq
from os.path import join, dirname
from dotenv import load_dotenv
    
from ollama import chat
from ollama import ChatResponse

class LLM:
    def __init__(self,):
        dotenv_path = join(dirname(__file__), '.env')
        load_dotenv(dotenv_path)
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def Groq_LLM(self,txt,word_limit = 50):
        prompt = f"Please provide a detailed yet concise response about the following topic, limited to {word_limit} words: {txt}"

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        return chat_completion.choices[0].message.content

    def ollama_LLM(self,txt,model='qwen2.5_1.5:latest',word_limit = 50):
        
        prompt = f"Please provide a detailed yet concise response about the following topic, limited to {word_limit} words: {txt}"

        response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        # print(response['message']['content'])
        # print(response.message.content)
        return response.message.content

if __name__=='__main__':
    llm = LLM()
    print(llm.ollama_LLM('AI'))