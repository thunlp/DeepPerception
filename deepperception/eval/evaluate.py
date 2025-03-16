import os
import re
import json
import argparse
import subprocess
import time
import torch

from datasets import load_dataset
from torchvision.ops.boxes import box_area
from multiprocessing import Process
from tqdm import tqdm


bbox_patterns = [
    re.compile(r'<answer>.*?\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)</answer>'),
    re.compile(r'So the answer is.*?\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\((.*?),.*?(.*?)\).*?\((.*?),.*?(.*?)\)'),
    re.compile(r'\[(\d*?), (\d*?), (\d*?), (\d*?)\]'),
    re.compile(r'\[(.*?), (.*?), (.*?), (.*?)\]'),
    re.compile(r'\((\d*?), (\d*?), (\d*?), (\d*?)\)'),
    re.compile(r'\((\d*?), (\d*?)\)\n?.*?\((\d*?), (\d*?)\)')
]

REF_PATTERN = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')
ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>')

def get_choice(ans):
    match = re.findall(ANSWER_PATTERN, ans)
    if len(match) > 0:
        choice = match[0].strip()
        if len(choice) > 1:
            choice = choice.split('.')[0]
        return choice
    else:
        return None
    
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def get_bbox(ans):
    for i, pattern in enumerate(bbox_patterns):
        predict_bbox = re.findall(pattern, ans)
        if len(predict_bbox) != 0:
            try:
                predict_bbox = (float(predict_bbox[-1][0].replace('[', '').replace('x', '')), float(predict_bbox[-1][1]), float(predict_bbox[-1][2]), float(predict_bbox[-1][3]))
            except:
                predict_bbox = [0, 0, 0, 0]
            if sum(predict_bbox) < 4:
                predict_bbox = [c*1000 for c in predict_bbox]

            return predict_bbox, i+1
    
    return (0., 0., 0., 0.), 0

def calculate_ious(category, results):
    ious = []
    correct = 0
    match_patterns_cnt = [0] * (len(bbox_patterns) + 1)
    for r in results:
        answer = r['answer']
        
        predict_bbox, i = get_bbox(answer)
        r['pred_bbox'] = predict_bbox
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)

        max_iou = 0
        for gt_bbox in r['gt_bbox']:
            target_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou > max_iou:
                max_iou = iou    
        
        ious.append(max_iou)
        r['iou'] = max_iou
        r['match pattern'] = i
        match_patterns_cnt[i] += 1
        if max_iou >= 0.5:
            correct += 1
    
    metrics = dict()
    acc = correct / len(ious)
    avg_iou = sum(ious)/len(ious)
    
    print(category)
    print(f'unmatch: {match_patterns_cnt[0]}, ' + ', '.join([f'match {i+1}: {cnt}' for i, cnt in enumerate(match_patterns_cnt[1:])]))    
    print(f'Acc @ 0.5: {acc}, IoU: {avg_iou}')
    
    metrics['all'] = {
        'Acc': acc,
        'IoU': avg_iou,
        'Num': len(ious)
    }
        
    return results, metrics

def eval(task, args, test_data):
    output_path = args.output_path
    
    all_metrics = dict()
    if task == 'grounding':
        seen_categories = args.seen_categories.split(',')
        all_categories = args.all_categories.split(',')
        
        
        results = {d: [] for d in all_categories}
        
        all_res = []
        seen_res = []
        unseen_res = []
        for data in tqdm(test_data):
            with open(f'{output_path}/temp/{data["question_id"]}.json', 'r') as f:
                r = json.load(f)
                results[data["category"]].append(r)
                all_res.append(r)
                if data["category"] in seen_categories:
                    seen_res.append(r)
                else:
                    unseen_res.append(r)

        all_res, metrics = calculate_ious('all', all_res)
        all_metrics['all'] = metrics
        
        seen_res, metrics = calculate_ious('seen domain', seen_res)
        all_metrics['seen domain'] = metrics
        
        unseen_res, metrics = calculate_ious('unseen domain', unseen_res)
        all_metrics['unseen domain'] = metrics
            
        for dataset, res in results.items():
            res, metrics = calculate_ious(dataset, res)
            all_metrics[dataset] = metrics
            with open(f'{args.output_path}/{dataset}.json', 'w') as f:
                json.dump(res, f, indent=4)
                
        with open(f'{args.output_path}/metrics.json', 'w') as f:
            json.dump(all_metrics, f)
            
    elif task == 'classification':
        correct = 0
        match_cnt = 0
        results = []
        for data in tqdm(test_data):
            with open(f'{output_path}/temp/{data["question_id"]}.json', 'r') as f:
                r = json.load(f)
            answer = r['answer']
            gt = data['messages'][1]['content']

            pred = get_choice(answer)
            
            if pred:
                match_cnt += 1
            if gt == pred:
                correct += 1
                r['correct'] = True
            else:
                r['correct'] = False
            results.append(r)
        
        acc = correct / len(test_data)
        print(f'Acc ({dataset}): {acc}')
        print(f'Match rate: {match_cnt/len(test_data)}')
        
        category = test_data['question_id'].split('/')[0]
        with open(f'{args.output_path}/{category}.json', 'w') as f:
            json.dump(results, f)
        with open(f'{args.output_path}/{category}-metrics.json', 'w') as f:
            json.dump(all_metrics, f)
    

def infer(args, json_path, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if args.vllm:
        subprocess.run(["python", 'inference.py',
                        "--data_path", args.data_path,
                        "--prompt", str(args.prompt),
                        "--vllm", 
                        "--id_path", json_path,
                        "--model_path", args.ckpt_path,
                        "--output_path", args.output_path])
    else:
        subprocess.run(["python", 'inference.py',
                        "--data_path", args.data_path,
                        "--prompt", str(args.prompt),
                        "--id_path", json_path,
                        "--model_path", args.ckpt_path,
                        "--output_path", args.output_path])

def launch_subprocesses(args, temp):
    processes = []
    temp_files = []
    
    if len(temp) > 0:
        if '72B' in args.ckpt_path:
            nprocs = args.num_processes
            if nprocs == 2:
                gpu_ids = ['0,1,2,3', '4,5,6,7']
            elif nprocs == 1:
                gpu_ids = [args.gpu_ids]
        else: # 7B-scale models
            gpu_ids = list(map(int, args.gpu_ids.split(',')))
            nprocs = len(gpu_ids)
        
        num_data_per_group = len(temp) // len(gpu_ids)
        
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * num_data_per_group
            end_idx = start_idx + num_data_per_group if i != (nprocs-1) else None
            
            timestamp = time.strftime("%Y%m%d%H%M%S")
            json_path = f'{args.output_path}/temp/{timestamp}_{gpu_id}.json'
            temp_files.append(json_path)
            with open(json_path, "w") as f:
                json.dump(temp[start_idx:end_idx], f)

            p = Process(target=infer, args=(args, json_path, gpu_id))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
            
        for temp_file in temp_files:
            os.remove(temp_file)

def get_data(args):
    output_path = args.output_path
    
    if args.data_path.endswith('.parquet'): # KVG-Bench
        task = 'grounding'
        all_categories = args.all_categories.split(',')
        for c in all_categories:
            os.makedirs(f'{output_path}/temp/{c}', exist_ok=True)
            
        dataset = load_dataset("parquet", data_files={"test": args.data_path})
        test_data = dataset['test']
    elif args.data_path.endswith('.json'): # FGVR
        task = 'classification'
        with open(args.data_path, 'r') as f:
            test_data = json.load(f)
        
        category = args.data_path.split('/')[-1].split('.')[0]
        os.makedirs(f'{output_path}/temp/{category}', exist_ok=True)
        i = 0
        for d in test_data:
            d['question_id'] = f'{category}/{str(i).zfill(5)}'
    else:
        print(f'No supported file type: {args.data_path}')
    
    qids = []
    for d in test_data:
        if not os.path.isfile(f'{output_path}/temp/{d["question_id"]}.json'):
            qids.append(d['question_id'])
    print(f'# Test data: {len(qids)}')
    
    return task, test_data, qids
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images across multiple GPUs.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt", required=False, default=None)
    parser.add_argument("--vllm", action='store_true')
    parser.add_argument("--seen_categories", required=False, default='aircraft,car,reptilia,bird,food')
    parser.add_argument("--all_categories", required=False, default='aircraft,car,reptilia,bird,food,dog,mollusca,mammal,flower,landmark')
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_processes", type=int, required=False, default=8)
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU IDs.")
    parser.add_argument("--output_path", required=True, help="Path to the output dir")

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print(f"Evaluating {args.ckpt_path}. Prompt: {args.prompt}. Results will be saved in {args.output_path}.")
    
    task, test_data, qids = get_data(args)
    launch_subprocesses(args, qids)
    eval(task, args, test_data)



if __name__ == "__main__":
    main()
