import os
import re
import json
import argparse
import subprocess
import random
import time
import torch
from tqdm import tqdm
import numpy as np

from PIL import Image
from datasets import load_dataset
from torchvision.ops.boxes import box_area
from multiprocessing import Process


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


bbox_patterns = [
    re.compile(r'<answer>.*?\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)</answer>'),
    re.compile(r'So the answer is.*?\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\((.*?),.*?(.*?)\).*?\((.*?),.*?(.*?)\)'),
    re.compile(r'\[(\d*?), (\d*?), (\d*?), (\d*?)\]'),
    re.compile(r'\[(.*?), (.*?), (.*?), (.*?)\]'),
    re.compile(r'\((\d*?), (\d*?), (\d*?), (\d*?)\)'),
    re.compile(r'\((\d*?), (\d*?)\)\n?.*?\((\d*?), (\d*?)\)'),
    re.compile(r'\[\[(\d*?),(\d*?),(\d*?),(\d*?)\]\]')
]

REF_PATTERN = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')
ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>')
QWEN3_JSON_PATTERN = re.compile(r'```json(.*?)```', re.DOTALL)

entity_patterns = [
    re.compile(r'<answer>.*?\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)</answer>'),
]


seen_train_entities_path = "./knowledge/all_entities_train.json"

visual_knowledge_path = "./knowledge/results-visual_knowledge.json"
with open(visual_knowledge_path, "r") as f:
    visual_knowledge_dataset = json.load(f)
entity_to_visual_knowledge = {}
for data_entity_name in visual_knowledge_dataset:
    entity_to_visual_knowledge[data_entity_name] = visual_knowledge_dataset[data_entity_name]["knowledge_category"]

visual_knowledge_ood_path = "./knowledge/results-visual_knowledge-trainood.json"
with open(visual_knowledge_ood_path, "r") as f:
    visual_knowledge_ood_dataset = json.load(f)
ood_entity_to_visual_knowledge = {}
for data_entity_name in visual_knowledge_ood_dataset:
    ood_entity_to_visual_knowledge[data_entity_name] = visual_knowledge_ood_dataset[data_entity_name]["knowledge_category"]

visual_knowledge_test_ood_path = "./knowledge/results-visual_knowledge-testood.json"
with open(visual_knowledge_test_ood_path, "r") as f:
    visual_knowledge_testood_dataset = json.load(f)
testood_entity_to_visual_knowledge = {}
for data_entity_name in visual_knowledge_testood_dataset:
    testood_entity_to_visual_knowledge[data_entity_name] = visual_knowledge_testood_dataset[data_entity_name]["knowledge_category"]



def get_choice(ans):
    if '<answer>' in ans:
        match = re.findall(ANSWER_PATTERN, ans)
        if len(match) > 0:
            ans = match[0]
        else:
            return None
    
    choice = ans.strip().lower()
    if len(choice) > 1:
        choice = choice.split('.')[0]
    return choice

    
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


def denorm_1000(box, W, H):
    """
    0~1000 -> pixel
    """
    if len(box) < 4:
        return [0, 0, 10, 10]

    x1 = box[0] / 1000 * W
    y1 = box[1] / 1000 * H
    x2 = box[2] / 1000 * W
    y2 = box[3] / 1000 * H
    return [x1, y1, x2, y2]

def denorm_wh(box):
    """
    [x, y, w, h] -> [x1, y1, x2, y2]
    """
    x, y, w, h = box

    x2 = x + w
    y2 = y + h
    return [x, y, x2, y2]

def repair_truncated_bbox_line(line: str):
    """
    修复 格式补全的 json 格式
    """
    # 去掉奇怪空白
    line = line.strip()

    # 必须是 bbox 行才修
    if '"bbox_2d"' not in line:
        return line

    # 1. 提取已经出现的数字
    nums = re.findall(r'\d+', line)

    # 2. 如果已经有 >=4 个数字，说明 bbox 数字齐了
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]

        # 3. 尝试提取 label（如果已经出现）
        label_match = re.search(r'"label"\s*:\s*"([^"]+)', line)
        if label_match:
            label = label_match.group(1)
        else:
            label_match = re.search(r'label"\s*:\s*"([^"]+)', line)
            if label_match:
                label = label_match.group(1)
            else:
                return line

        # 4. 构造完整合法 JSON
        fixed = {
            "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
            "label": label
        }

        return json.dumps(fixed)

    # 数字都不够，直接返回原始
    return line


def match_and_count(pred_boxes, gt_boxes, iou_thr=0.5):
    matched_gt = set()
    tp = 0  # 预测情况的正例

    for p in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, g in enumerate(gt_boxes):
            if i in matched_gt:  # 避免算进去重合的框
                continue

            p = torch.tensor(p, dtype=torch.float32).view(-1, 4)
            g = torch.tensor(g, dtype=torch.float32).view(-1, 4)

            iou, _ = box_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


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

def calculate_ious(category, results, args):
    ious = []
    correct = 0
    match_patterns_cnt = [0] * (len(bbox_patterns) + 1)
    for rr in results:
        r = rr['result']
        answer = r['answer']
        match = re.findall(ANSWER_PATTERN, answer)
        if len(match) > 0:
            answer = match[0].strip()

        if '```json' in answer:
            match = re.search(QWEN3_JSON_PATTERN, answer)
            if match:
                answer = match.group(1)
        
        predict_bbox, i = get_bbox(answer)
        r['pred_bbox'] = predict_bbox
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)

        max_iou = 0
        for gt_bbox in r['gt_bbox']:
            # if args.infer_model == 'glm-4_1v':
            #     h, w = r['hw']
            #     gt_bbox = [gt_bbox[0]/1000*w, gt_bbox[1]/1000*h, gt_bbox[2]/1000*w, gt_bbox[3]/1000*h]
            #     r['gt_bbox_rawimgae'] = gt_bbox
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
    
    print("----------------")
    print(category)
    print("----------------")
    print(f'unmatch: {match_patterns_cnt[0]}, ' + ', '.join([f'match {i+1}: {cnt}' for i, cnt in enumerate(match_patterns_cnt[1:])]))    
    print(f'Acc @ 0.5: {acc}, IoU: {avg_iou}')
    
    metrics['all'] = {
        'Acc': acc,
        'IoU': avg_iou,
        'Num': len(ious)
    }
        
    return results, metrics

def calculate_visual_knowledge_ious(category, results, args):
    """
    计算 id 和 ood 的 各个 visual_knowledge 的部分
    """
    # ======================
    # 划分 id 和 ood 的结果
    # ======================

    with open(seen_train_entities_path, "r") as f:
        train_seen_entities_and_id = json.load(f)
    train_seen_entities = [seen_i['entity_name'] for seen_i in train_seen_entities_and_id]

    id_results = []
    ood_results = []
    ood_entities = []
    ood_entities_cases = []

    for rr in results:
        r = rr['result']
        answer = r['answer']
        entity_name = rr['entity_name']
        if entity_name in train_seen_entities:
            id_results.append(rr)
        else:
            ood_results.append(rr)
            # if entity_name not in entity_to_visual_knowledge:
            #     ood_entities_cases.append(entity_name)
            #     if entity_name not in ood_entities:
            #         # print(entity_name)
            #         ood_entities.append(entity_name)

    # print(f"----------------")
    # print(f"{len(ood_entities_cases)} / {len(results)}")
    # print(f"----------------")


    def cal_knowledge_ious(category, results, level, mode="direct_average", data_version="id"):
        ious = []
        correct = 0
        knowledge_corrects = {}
        knowledge_totals = {}
        match_patterns_cnt = [0] * (len(bbox_patterns) + 1)

        if data_version == "id":
            entity_to_visual_knowledge_cur = entity_to_visual_knowledge
        elif data_version == "ood":
            entity_to_visual_knowledge_cur = ood_entity_to_visual_knowledge

        for rr in results:
            r = rr['result']
            answer = r['answer']
            match = re.findall(ANSWER_PATTERN, answer)
            if len(match) > 0:
                answer = match[0].strip()
            entity_name = rr['entity_name']

            correct_category = entity_to_visual_knowledge_cur[entity_name]
            if correct_category not in knowledge_corrects:
                knowledge_corrects[correct_category] = 0
                knowledge_totals[correct_category] = 0

            knowledge_totals[correct_category] += 1

            if '```json' in answer:
                match = re.search(QWEN3_JSON_PATTERN, answer)
                if match:
                    answer = match.group(1)
            
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
            r['correct_category'] = correct_category
            match_patterns_cnt[i] += 1
            if max_iou >= 0.5:
                correct += 1
                knowledge_corrects[correct_category] += 1

        assert len(ious) == len(results)
        
        metrics = dict()
        acc = correct / len(ious)
        avg_iou = sum(ious)/len(ious)

        for cate in knowledge_corrects:
            avg_acc_knowledge = knowledge_corrects[cate] / knowledge_totals[cate]
            print(f'VisualKN  {level} {cate} Acc @ 0.5: {avg_acc_knowledge} ({knowledge_corrects[cate]}/{knowledge_totals[cate]})')
            metrics[cate] = {
                'Acc': avg_acc_knowledge,
                'Num': knowledge_totals[cate]
            }

        if mode == "direct_average":
            metrics['total'] = {
                'Acc': acc,
                'Num': len(results)
            }
            print(f'VisualKN  {level} average Acc @ 0.5: {acc} ({correct}/{len(results)})')

        return results, metrics


    id_results, id_metrics = cal_knowledge_ious(category, id_results, level="in domain", mode="direct_average", data_version="id")
    ood_results, ood_metrics = cal_knowledge_ious(category, ood_results, level="out of domain", mode="direct_average", data_version="ood")

    metrics = dict()
    metrics['in domain'] = id_metrics
    metrics['out of domain'] = ood_metrics

    return id_results + ood_results, metrics

def calculate_visual_knowledge_testood_ious(category, results, args):
    """
    计算 test unseen 的 各个 visual_knowledge 的部分
    """
    def cal_knowledge_ious(category, results, level, mode="direct_average"):
        ious = []
        correct = 0
        knowledge_corrects = {}
        knowledge_totals = {}
        match_patterns_cnt = [0] * (len(bbox_patterns) + 1)

        entity_to_visual_knowledge_cur = testood_entity_to_visual_knowledge

        for rr in results:
            r = rr['result']
            answer = r['answer']
            match = re.findall(ANSWER_PATTERN, answer)
            if len(match) > 0:
                answer = match[0].strip()
            entity_name = rr['entity_name']

            correct_category = entity_to_visual_knowledge_cur[entity_name]
            if correct_category not in knowledge_corrects:
                knowledge_corrects[correct_category] = 0
                knowledge_totals[correct_category] = 0

            knowledge_totals[correct_category] += 1

            if '```json' in answer:
                match = re.search(QWEN3_JSON_PATTERN, answer)
                if match:
                    answer = match.group(1)
            
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
            r['correct_category'] = correct_category
            match_patterns_cnt[i] += 1
            if max_iou >= 0.5:
                correct += 1
                knowledge_corrects[correct_category] += 1

        assert len(ious) == len(results)
        
        metrics = dict()
        acc = correct / len(ious)
        avg_iou = sum(ious)/len(ious)

        for cate in knowledge_corrects:
            avg_acc_knowledge = knowledge_corrects[cate] / knowledge_totals[cate]
            print(f'VisualKN  {level} {cate} Acc @ 0.5: {avg_acc_knowledge} ({knowledge_corrects[cate]}/{knowledge_totals[cate]})')
            metrics[cate] = {
                'Acc': avg_acc_knowledge,
                'Num': knowledge_totals[cate]
            }

        if mode == "direct_average":
            metrics['total'] = {
                'Acc': acc,
                'Num': len(results)
            }
            print(f'VisualKN  {level} average Acc @ 0.5: {acc} ({correct}/{len(results)})')

        return results, metrics

    ood_results, ood_metrics = cal_knowledge_ious(category, results, level="out of domain", mode="direct_average")

    metrics = dict()
    metrics['out of domain'] = ood_metrics

    return ood_results, metrics

def calculate_high_low_knowledge_metrics(category, metrics_knowledge, args):
    high_knowledge = ['completely correct', 'mostly correct', 'partially correct']
    low_knowledge = ['mostly incorrect', 'completely incorrect']

    metrics_save = {}
    highKN_nums = 0
    highKN_correct_nums = 0
    lowKN_nums = 0
    lowKN_correct_nums = 0

    if "in domain" in metrics_knowledge:
        in_domain_metrics = metrics_knowledge["in domain"]
        for k in in_domain_metrics:
            if k == 'total':
                continue
            if k in high_knowledge:
                highKN_nums += in_domain_metrics[k]['Num']
                highKN_correct_nums += int(in_domain_metrics[k]['Num'] * in_domain_metrics[k]['Acc'])
            if k in low_knowledge:
                lowKN_nums += in_domain_metrics[k]['Num']
                lowKN_correct_nums += int(in_domain_metrics[k]['Num'] * in_domain_metrics[k]['Acc'])
                

    if "out of domain" in metrics_knowledge:
        out_of_domain_metrics = metrics_knowledge["out of domain"]
        for k in out_of_domain_metrics:
            if k == 'total':
                continue
            if k in high_knowledge:
                highKN_nums += out_of_domain_metrics[k]['Num']
                highKN_correct_nums += int(out_of_domain_metrics[k]['Num'] * out_of_domain_metrics[k]['Acc'])
            if k in low_knowledge:
                lowKN_nums += out_of_domain_metrics[k]['Num']
                lowKN_correct_nums += int(out_of_domain_metrics[k]['Num'] * out_of_domain_metrics[k]['Acc'])



    metrics_save = {}
    metrics_save["high knowledge"] = {
        'Acc': highKN_correct_nums / highKN_nums,
        'Correct': highKN_correct_nums,
        'Num': highKN_nums
    }
    metrics_save["low knowledge"] = {
        'Acc': lowKN_correct_nums / lowKN_nums,
        'Correct': lowKN_correct_nums,
        'Num': lowKN_nums
    }
    metrics_save["total"] = {
        'Acc': (highKN_correct_nums+lowKN_correct_nums) / (highKN_nums+lowKN_nums),
        'Num': highKN_nums+lowKN_nums
    }


    print(f"{category} high_knowledge:  acc: {highKN_correct_nums / highKN_nums},  num: {highKN_nums}")
    print(f"{category} low_knowledge:  acc: {lowKN_correct_nums / lowKN_nums},  num: {lowKN_nums}")
    print(f"{category} total:  acc: {(highKN_correct_nums+lowKN_correct_nums) / (highKN_nums+lowKN_nums)},  num: {highKN_nums+lowKN_nums}")
    
    return metrics_save

def calculate_high_low_knowledge_metrics_all(seen_metrics_high_low_knowledge, unseen_metrics_high_low_knowledge):
    high_knowledge = ['completely correct', 'mostly correct', 'partially correct']
    low_knowledge = ['mostly incorrect', 'completely incorrect']

    metrics_save = {}
    highKN_nums = seen_metrics_high_low_knowledge["high knowledge"]['Num'] + unseen_metrics_high_low_knowledge["high knowledge"]['Num']
    highKN_correct_nums = seen_metrics_high_low_knowledge["high knowledge"]['Correct'] + unseen_metrics_high_low_knowledge["high knowledge"]['Correct']
    lowKN_nums = seen_metrics_high_low_knowledge["low knowledge"]['Num'] + unseen_metrics_high_low_knowledge["low knowledge"]['Num']
    lowKN_correct_nums = seen_metrics_high_low_knowledge["low knowledge"]['Correct'] + unseen_metrics_high_low_knowledge["low knowledge"]['Correct']

    metrics_save["high knowledge"] = {
        'Acc': highKN_correct_nums / highKN_nums,
        'Correct': highKN_correct_nums,
        'Num': highKN_nums
    }
    metrics_save["low knowledge"] = {
        'Acc': lowKN_correct_nums / lowKN_nums,
        'Correct': lowKN_correct_nums,
        'Num': lowKN_nums
    }
    metrics_save["total"] = {
        'Acc': (highKN_correct_nums+lowKN_correct_nums) / (highKN_nums+lowKN_nums),
        'Num': highKN_nums+lowKN_nums
    }

    print("=====================")
    print(f"ALL high_knowledge:  acc: {highKN_correct_nums / highKN_nums},  num: {highKN_nums}")
    print(f"ALL low_knowledge:  acc: {lowKN_correct_nums / lowKN_nums},  num: {lowKN_nums}")
    print(f"ALL total:  acc: {(highKN_correct_nums+lowKN_correct_nums) / (highKN_nums+lowKN_nums)},  num: {highKN_nums+lowKN_nums}")
    print("=====================")


    return metrics_save


FINAL_CUES = [
    "final answer",
    "the answer is",
    "answer is",
    "i choose",
    "i select",
]

def has_final_cue(text):
    text = text.lower()
    return any(cue in text for cue in FINAL_CUES)



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
                r['question'] = data['question']
                match = re.search(REF_PATTERN, data['answer'])
                answer_entity = match.group(1)
                results[data["category"]].append(
                    {
                        "result": r,
                        "entity_name": answer_entity
                    }
                )
                all_res.append(
                    {
                        "result": r,
                        "entity_name": answer_entity
                    }
                )
                if data["category"] in seen_categories:
                    seen_res.append(
                        {
                            "result": r,
                            "entity_name": answer_entity
                        }
                    )
                else:
                    unseen_res.append(
                        {
                            "result": r,
                            "entity_name": answer_entity
                        }
                    )

        all_res, metrics = calculate_ious('all', all_res, args)
        
        all_metrics['all'] = metrics
        
        _, metrics = calculate_ious('seen domain', seen_res, args)
        _, seen_metrics_visual_knowledge = calculate_visual_knowledge_ious('seen domain', seen_res, args)
        seen_metrics_high_low_knowledge = calculate_high_low_knowledge_metrics('seen domain', seen_metrics_visual_knowledge, args)
        all_metrics['seen domain'] = metrics
        all_metrics['seen domain']['high_low_knowledge'] = seen_metrics_high_low_knowledge
        
        _, metrics = calculate_ious('unseen domain', unseen_res, args)
        _, unseen_metrics_visual_knowledge = calculate_visual_knowledge_testood_ious('unseen domain', unseen_res, args)
        unseen_metrics_high_low_knowledge = calculate_high_low_knowledge_metrics('unseen domain', unseen_metrics_visual_knowledge, args)
        all_metrics['unseen domain'] = metrics
        all_metrics['unseen domain']['high_low_knowledge'] = unseen_metrics_high_low_knowledge

        # caluate the results of highKN and lowKN
        all_metrics_high_low_knowledge = calculate_high_low_knowledge_metrics_all(seen_metrics_high_low_knowledge, unseen_metrics_high_low_knowledge)
        all_metrics['all']['high_low_knowledge'] = all_metrics_high_low_knowledge
            
        for dataset, res in results.items():
            res, metrics = calculate_ious(dataset, res, args)
            all_metrics[dataset] = metrics

            with open(f'{args.output_path}/{dataset}.json', 'w') as f:
                json.dump(res, f, indent=4)
                
        with open(f'{args.output_path}/metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)
    else:
        raise ValueError(f"Unsupported task: {task}")

def infer(args, json_path, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if args.vllm:
        subprocess.run(["python", 'inference.py',
                        "--data_path", args.data_path,
                        "--prompt", str(args.prompt),
                        "--vllm", 
                        "--id_path", json_path,
                        "--model_path", args.ckpt_path,
                        "--output_path", args.output_path,
                        "--infer_model", args.infer_model,
                        "--seed", str(args.seed)])
    else:
        subprocess.run(["python", 'inference.py',
                        "--data_path", args.data_path,
                        "--prompt", str(args.prompt),
                        "--id_path", json_path,
                        "--model_path", args.ckpt_path,
                        "--output_path", args.output_path,
                        "--infer_model", args.infer_model,
                        "--seed", args.seed])

def launch_subprocesses(args, temp):
    processes = []
    temp_files = []
    
    if len(temp) > 0:
        if '72B-' in args.ckpt_path:
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
    else:
        print(f'No supported file type: {args.data_path}')
    
    qids = []
    for d in tqdm(test_data):
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_processes", type=int, required=False, default=8)
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU IDs.")
    parser.add_argument("--output_path", required=True, help="Path to the output dir")
    parser.add_argument("--infer_model", required=True, choices=['qwen2_vl', 'qwen3_vl'], help="The model used to inference")

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print(f"Evaluating {args.ckpt_path}. Prompt: {args.prompt}. Results will be saved in {args.output_path}.")
    
    task, test_data, qids = get_data(args)
    launch_subprocesses(args, qids)
    eval(task, args, test_data)



if __name__ == "__main__":
    args = parse_arguments()

    set_seed(args.seed)

    main()
