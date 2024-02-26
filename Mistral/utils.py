import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig
)
from langchain.llms import HuggingFacePipeline


def build_llm_pipeline(model_name, max_new_tokens, temperature, repetition_penalty):
    """
    Function to build langchain huggingface pipeline to define llm for our few shot classification tasks
    Params:
        model_name (str) : Hugging Face model to be used (e.g. mistralai/Mistral-7B-Instruct-v0.2)
        max_new_tokens (int) : 
        temperature (float) :
        repetition_penalty (float) : 
    Returns:
        HuggingFacePipeline mistral llm object
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set up quantization config
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
    
    # Load pre-trained config
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                quantization_config=bnb_config)

    # Model pipeline
    model_pipeline = pipeline(model=model,
                            tokenizer=tokenizer,
                            task="text-generation",
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=max_new_tokens,
                            do_sample = temperature > 0.0)                                           

    llm = HuggingFacePipeline(pipeline=model_pipeline)

    return llm