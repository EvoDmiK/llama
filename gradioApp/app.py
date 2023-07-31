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
def llama(prompt, temperature, top_p):
    
    LOGGER.info(f'Q. {prompt}')
    dialogs = [[{
                "role"    : "user",
                "content" : prompt 
            }]]
    
    ## 입력받은 prompt로 텍스트 생성 
    LOGGER.info('Receieve prompt and generating answers...')
    results = generator.chat_completion(dialogs, max_gen_len = None,
                                        temperature = temperature, top_p = top_p)
    
    ## 생성한 결과물들 띄어쓰기로 묶음.
    result = ' \n'.join([res["generation"]["content"] for res in results])
    
    ## 생성한 결과물 로깅 및 반환
    LOGGER.info(f'A. {result}')
    return f'A. {result}'


## Input, Output 텍스트 박스 각 하나씩에, Temperture, Top_p 슬라이더 바 각 하나씩
app = gr.Interface(fn = llama, inputs = ['text', 
                                         gr.Slider(0, 1, step = 0.01, label = 'Temperature'), 
                                         gr.Slider(0, 1, step = 0.01, label = 'Top_p')], 
                                        outputs = 'text')

## 60초 Timeout을 방지하기 위한 queue() 추가
app.queue().launch(share = True, inline = False, debug = True)