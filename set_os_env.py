import os

def set_os_env():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ['HF_DATASETS_OFFLINE '] = "1"
