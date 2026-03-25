import argparse
import io
import os
import re
import json
import time
import torch
import base64
import random
import numpy as np

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


PATTERN = re.compile(r'<\|box_start\|>\(([0-9]*?),([0-9]*?)\),\(([0-9]*?),([0-9]*?)\)<\|box_end\|>')
REF_PATTERN = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')

GROUNDING_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (bounding box) in <answer> </answer> tags."
R1_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt", required=False, default=None)
    parser.add_argument("--id_path", required=True)
    parser.add_argument("--model_path", required=True, help="Path to qwen.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--infer_model", required=True, help="The model used to inference")
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument("--seed", required=True)

    parser.add_argument("--batch_size", required=False, type=int, default=1)

    args = parser.parse_args()

    return args



def qwenvl_inference(model, processor, messages, infer_model):
    if infer_model == 'qwen2_vl':
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
    elif infer_model == "qwen3_vl":
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages, image_patch_size=16)  # processor.image_processor.patch_size == 16

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    generated_text = response[0]

    return generated_text


def inference_grounding(model, processor, sampling_params, prompt, query, image_bytes, infer_model, generation_config):
    encoded_string = 'data:image:base64,' + str(base64.b64encode(image_bytes).decode("utf-8"))
    messages = []
    cot_response = None
    # CoT
    if prompt == 'cot-normal':
        query += ". Let's think step by step"
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    elif prompt == 'r1':
        query = GROUNDING_TEMPLATE.format(Question=query)
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    elif prompt == 'qwen3-vl':
        # print(query)
        query += ", output its bbox coordinates using JSON format."
        # query: 
        # Find and give the bounding box of <|object_ref_start|>Sukhoi Su-30<|object_ref_end|>, output its bbox coordinates using JSON format.
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})
    else:
        messages.append({"role": "user", "content": [dict(type='image_url', image_url=encoded_string), dict(type='text', text=query)]})

    print(query)

    # Grounding
    if sampling_params:
        if infer_model == 'qwen2_vl':
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
        elif infer_model == "qwen3_vl":
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages, image_patch_size=16)  # processor.image_processor.patch_size == 16

        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        outputs = model.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
    else:
        if infer_model == 'qwen2_vl' or infer_model == 'qwen3_vl':
            generated_text = qwenvl_inference(model, processor, messages, infer_model)
        else:
            raise ValueError(f"Unsupported model: {infer_model}")
    
    return {
        'cot': cot_response,
        'answer': generated_text,
    }


def infer(model, processor, sampling_params, args, generation_config):
    prompt = args.prompt
    output_path = args.output_path
    
    if args.data_path.endswith('.parquet'): # KVG-Bench
        task = 'grounding'
        dataset = load_dataset("parquet", data_files={"test": args.data_path})
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
            if '<image>' not in query:
                query = f'<image>{query}'

            image_bytes = data['image']['bytes']
            gt = data['answer']
            match = re.search(PATTERN, gt)
            bbox = [[float(match[1]), float(match[2]), float(match[3]), float(match[4])]]
            
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            
            out_filename = f"{output_path}/temp/{data['question_id']}.json"
            
            response = inference_grounding(model, processor, sampling_params, prompt, query, image_bytes, args.infer_model, generation_config)
            response['gt_bbox'] = bbox
            response['hw'] = (h ,w)
        else:
            raise ValueError(f"Unsupported task: {task}")
            
            
        with open(out_filename, 'w') as f:
            json.dump(response, f)
        

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    processor = None
    generation_config = None

    if args.vllm:
        from vllm import LLM, SamplingParams
        num = torch.cuda.device_count()
        print(f"Num visiable gpu: {num}")
        model = LLM(args.model_path, max_model_len=17920, tensor_parallel_size=num, gpu_memory_utilization=0.6)
        sampling_params = SamplingParams(n=1, temperature=0.5, max_tokens=4096)
        # sampling_params = SamplingParams(n=1, temperature=0.5, max_tokens=4096)

    else:
        sampling_params = None

        if args.infer_model == 'qwen2_vl':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
    
    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model_path)

    start_time = time.time()
    
    with torch.no_grad():
        infer(model, processor, sampling_params, args, generation_config)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Evaluate Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    args = parse_args()
    set_seed(int(args.seed))

    main()