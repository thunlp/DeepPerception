import argparse
import io
import os
import re
import json
import time
import torch
import base64

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PATTERN = re.compile(r'<\|box_start\|>\(([0-9]*?),([0-9]*?)\),\(([0-9]*?),([0-9]*?)\)<\|box_end\|>')
REF_PATTERN = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')

GROUNDING_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (bounding box) in <answer> </answer> tags."
# QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (bounding box in (x1,y1),(x2,y2) format) in <answer> </answer> tags."
CLASSIFICATION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt", required=False, default=None)
    parser.add_argument("--id_path", required=True)
    parser.add_argument("--model_path", required=True, help="Path to qwen.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument('--vllm', action='store_true')

    parser.add_argument("--batch_size", required=False, type=int, default=1)

    args = parser.parse_args()

    return args

def inference_classification(model, processor, sampling_params, prompt, query, image):
    messages = []
    
    if prompt == 'r1':
        query = CLASSIFICATION_TEMPLATE.format(Question=query)
        messages.append({"role": "user", "content": [dict(type='image', image=image), dict(type='text', text=query)]})
    else:
        messages.append({"role": "user", "content": [dict(type='image', image=image), dict(type='text', text=query)]})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    if sampling_params:
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        outputs = model.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
    else:
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=1500)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_text = response[0]
    
    return {
        'answer': generated_text,
    }

def inference_grounding(model, processor, sampling_params, prompt, query, image_bytes):
    encoded_string = 'data:image:base64,' + str(base64.b64encode(image_bytes).decode("utf-8"))
    messages = []
    cot_response = None
    # CoT
    if prompt == 'cot-kvg':
        match = re.search(REF_PATTERN, query)
        ref = match[1]
        
        cot_text = (
            f'Which object in this image is {ref}? Give a detailed and discriminative description of the appearance of it'
        )
        
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=cot_text)]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        if sampling_params:
            llm_inputs = {
                "prompt": text,
                "multi_modal_data": {
                    "image": image_inputs
                }
            }
            outputs = model.generate([llm_inputs], sampling_params=sampling_params)
            cot_response = outputs[0].outputs[0].text
        else:
            inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=1500)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            cot_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        messages.append({"role": "assistant", "content": cot_response})
        grounding_text = f'Based on the description, find and give the bounding box of <|object_ref_start|>{ref}<|object_ref_end|>'
        messages.append({"role": "user", "content": [dict(type='text', text=grounding_text)]})
    elif prompt == 'cot-normal':
        query += ". Let's think step by step"
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    elif prompt == 'r1':
        query = GROUNDING_TEMPLATE.format(Question=query)
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    else:
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    
    # Grounding
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    if sampling_params:
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        outputs = model.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
    else:
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=1500)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_text = response[0]
    
    return {
        'cot': cot_response,
        'answer': generated_text,
    }

def infer(model, processor, sampling_params, args):
    prompt = args.prompt
    output_path = args.output_path
    
    if args.data_path.endswith('.parquet'): # KVG-Bench
        task = 'grounding'
        dataset = load_dataset("parquet", data_files={"test": args.data_path})
    elif args.data_path.endswith('.json'): # FGVR
        task = 'classification'
        with open(args.data_path, 'r') as f:
            dataset = json.load(f)
    else:
        print(f'No supported file type: {args.data_path}')
    
    with open(args.id_path, 'r') as f:
        qids = json.load(f)
        
    test_data = []
    for d in dataset['test']:
        if d['question_id'] in qids:
            test_data.append(d)
    
    for data in tqdm(test_data):
        if task == 'grounding':
            query = data['question']
            image_bytes = data['image']['bytes']
            
            gt = data['answer']
            match = re.search(PATTERN, gt)
            bbox = [[float(match[1]), float(match[2]), float(match[3]), float(match[4])]]
            
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            
            out_filename = f"{output_path}/temp/{data['question_id']}.json"
            
            response = inference_grounding(model, processor, sampling_params, prompt, query, image_bytes)
            response['gt_bbox'] = bbox
            response['hw'] = (h ,w)
        elif task == 'classification':
            query = data['messages'][0]['content'].replace('<image>', '')
            image = data['images'][0]
            
            
            out_filename = f"{output_path}/temp/{data['question_id']}.json"
            response = inference_grounding(model, processor, sampling_params, prompt, query, image)
            
            
        with open(out_filename, 'w') as f:
            json.dump(response, f)
        

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    if args.vllm:
        from vllm import LLM, SamplingParams
        model = LLM(args.model_path, max_model_len=17920, tensor_parallel_size=1)
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=1536)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        sampling_params = None
    
    processor = AutoProcessor.from_pretrained(args.model_path)

    start_time = time.time()
    
    with torch.no_grad():
        infer(model, processor, sampling_params, args)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Evaluate Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()