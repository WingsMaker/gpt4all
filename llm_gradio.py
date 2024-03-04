import gradio as gr
from gpt4all import GPT4All
from pathlib import Path

model = GPT4All(model_name='gpt4all-falcon-newbpe-q4_0.gguf',
                model_path=(Path.home() / '.cache' / 'gpt4all'),
                allow_download=False)    

def ask_falcon(message, history):
    response = model.generate(message, temp=0)
    print(response)
    return response
    
gr.ChatInterface(
    ask_falcon,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=8),
    title="LLM Demo",
    description="LLM Demo using offline mode",
    theme="soft",
    examples=None,
    cache_examples=[],
    retry_btn=None,
    undo_btn=None,
    clear_btn=None,
).launch()
