import os

import gradio as gr

## misc/utils.py 파일에 있는 build, main, LOGGER 사용
from misc.utils import *

## 경로 분할자 (path seperator) 지정 (OS에 따라 달라 os.path.sep 사용 -> 우분투 : /, 윈도우 : \)
SEP            = os.path.sep 
ROOT_PATH      = SEP.join(os.getcwd().split(SEP)[:-1])

## 체크포인트, 토크나이저 경로
CKPT_PATH      = f'{ROOT_PATH}/checkpoints/7B-chat'
MODEL_PATH     = f'{CKPT_PATH}/llama-2-7b-chat'
TOKENIZER_PATH = f'{CKPT_PATH}/tokenizer.model'

## 텍스트 generator 생성
LOGGER.info('Generator is loading...')
generator = build(MODEL_PATH, TOKENIZER_PATH, 0, 1)
LOGGER.info('Generator loading is complete.')

## gradio에서 사용할 함수 생성
def llama(prompt):
    
    LOGGER.info(f'Q. {prompt}')
    dialogs = [[{
                "role"    : "user",
                "content" : prompt 
            }]]
    
    ## 입력받은 prompt로 텍스트 생성 
    LOGGER.info('Receieve prompt and generating answers...')
    results = generator.chat_completion(dialogs, max_gen_len = None,
                                        temperature = 0.8, top_p = 0.95)
    
    ## 생성한 결과물들 띄어쓰기로 묶음.
    result = ' \n'.join([res["generation"]["content"] for res in results])
    
    ## 생성한 결과물 로깅 및 반환
    LOGGER.info(f'A. {result}')
    return f'A. {result}'


## 입력이랑 출력을 chatbot 창에 로깅해주는 함수
def prompt_and_history(prompt, history):
    
    history = history or []
    input_  = ''.join(list(prompt))
    result  = llama(input_)
    
    history.append((prompt, result))
    return history, history


app = gr.Blocks(theme = gr.themes.Monochrome())
with app:
    gr.Markdown("""
                    <h1><center>LlaMA ChatBot with Gradio and Meta</center></h1>
                """)
    
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder = "Enter your query here")
    state   = gr.State()
    submit  = gr.Button("SEND")
    
    submit.click(prompt_and_history,
                 inputs  = [message, state],
                 outputs = [chatbot, state])

## 60초 Timeout을 방지하기 위한 queue() 추가
app.queue().launch(share = True, inline = False, debug = True)