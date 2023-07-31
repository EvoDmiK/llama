from typing import Tuple, List, Dict
import json
import time
import sys
import gc
import os

import torch

## 상위 폴더에 있는 모듈들 임포트 시킬 수 있도록 세팅
SEP       = os.path.sep 
ROOT_PATH = SEP.join(os.getcwd().split(SEP)[:-1])
sys.path.append(ROOT_PATH)

## GPU 비병렬 처리용 코드를 사용하여 구현 (그래서 오래걸림. 큐에 하나만 쌓여있으면 평균 4 ~ 5분,,,)
from non_parallel_llama.model import ModelArgs, Transformer
from non_parallel_llama.tokenizer import Tokenizer
from non_parallel_llama import Llama

## misc/logger.py 코드에서 logger 가져오는 함수
from misc.logger import get_logger

## 메모리 비우도록 하는 코드
torch.cuda.empty_cache()
gc.collect()

LOGGER         = get_logger()

def build(ckpt_path: str, tokenizer_path: str, local_rank: int = 0, world_size: int = 1) -> Llama:
    
    ## 체크포인트 경로에서 '.pth'파일이 포함되어 있는 녀석만 가져오기
    checkpoints = sorted([path for path in os.listdir(ckpt_path)
                          if '.pth' in path])
    
    if world_size != len(checkpoints):
        mesg = f'Loading a checkpoint for MP = {len(checkpoints)} but world_size is {world_size}'
        LOGGER.error(mesg)
    
    ## 체크포인트 로딩
    ckpt       = f'{ckpt_path}/{checkpoints[local_rank]}'
    ckpt       = torch.load(ckpt)
    params     = json.loads(open(f'{ckpt_path}/params.json', 'r').read())
    model_args = ModelArgs(max_seq_len = 1024, max_batch_size = 32, **params)
    tokenizer  = Tokenizer(model_path = tokenizer_path)
    
    model_args.vocab_size = tokenizer.n_words
    model                 = Transformer(model_args)
    model.load_state_dict(ckpt, strict = False)
    generator             = Llama(model, tokenizer)
    
    return generator  
    
    

def main(generator: Llama, prompt: str,
         temperature: float = 0.8, top_p: float = 0.95) -> List[Dict]:
    
    dialog    = [[{
                    "role"    : "user",
                    "content" : prompt 
                }]]
    
    LOGGER.info(f'Q. {prompt}')
    results = generator.chat_completion(dialog, max_gen_len = None,
                                        temperature = temperature, top_p = top_p)
    
    return results