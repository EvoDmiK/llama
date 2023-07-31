import logging

def get_logger():
    
    logger = logging.getLogger('Gradio App')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s : %(message)s')
    handler   = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger