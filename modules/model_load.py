from auto_gptq import AutoGPTQForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from shared import model_name
import shared

model_name_or_path = f"models\{model_name}"

def load_model():

    params = {
            'model_basename': 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order', 
            'device': 'cuda:0', 
            'use_triton': False, 
            'inject_fused_attention': True, 
            'inject_fused_mlp': True, 
            'use_safetensors': True, 
            'trust_remote_code': False, 
            'max_memory': {0: '11GiB', 'cpu': '99GiB'}, 
            'quantize_config': None, 
            'use_cuda_fp16': True
        }

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, **params)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    shared.model = model
    shared.tokenizer = tokenizer

    return model, tokenizer