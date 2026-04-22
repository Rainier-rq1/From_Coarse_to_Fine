from vllm import LLM, SamplingParams
import os
import json
import re
import time
from collections import defaultdict
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def extract_score(text):
    match = re.search(r'\[\[(\d+)\]\]', text)
    if match:
        try:
            return int(match.group(1))
        except:
            return 0
    return 0

def process_question(item):
    response = item.get('chosen')
    constraints = item.get('constraint', [])

    prompts = []
    for checker in constraints:
        prompt = f'''请判断给定的回复是否遵循约束，比如长度、风格、格式等约束。
[回复]
{response}
[约束]
{checker}
请判断给定的回复是否遵循约束，比如长度、风格、格式等约束，请在回答的最开始用[[score]]格式输出你的分数。如果遵循约束，请输出[[1]]，否则输出[[0]]'''
        prompts.append(prompt)

    return prompts

def generate_messages(data, batch_size):
    all_messages = []

    for item_index, item in enumerate(data):
        prompts = process_question(item)
        for idx, prompt in enumerate(prompts):
            message = {
                "item_index": item_index,
                "constraint_index": idx,
                "message": [{"role": "user", "content": prompt}]
            }
            all_messages.append(message)

    all_batches = []
    num_batches = (len(all_messages) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(all_messages))
        batch = all_messages[start:end]
        all_batches.append(batch)

    return all_batches, len(all_batches)

def infer_group_size_from_filename(filename):
    """从文件名推断group_size（3c, 4c, 5c）"""
    basename = os.path.basename(filename)
    # 匹配3c, 4c, 5c模式
    match = re.search(r'(\d+)c', basename)
    if match:
        return int(match.group(1))
    # 默认值
    return 3

def main():
    model_path = "models--Qwen2.5-7B-Instruct"
    model = LLM(model_path, dtype='bfloat16', gpu_memory_utilization=0.8, tensor_parallel_size=4, max_num_seqs=256)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_file = "eval.jsonl"
    output_file = "llm_judge_ranked.jsonl"

    # 自动推断group_size
    group_size = infer_group_size_from_filename(input_file)
    print(f"检测到输入文件: {os.path.basename(input_file)}")
    print(f"自动设置 group_size = {group_size}")

    # Load input data
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    batch_size = 1000
    messages_all, num_batches = generate_messages(data, batch_size)

    item_scores = defaultdict(list)
    item_times = {}  # 记录每个 item 的耗时
    item_start_flags = {}  # 用于标记开始计时的 item

    for batch_idx in range(num_batches):
        batch = messages_all[batch_idx]
        prompts = [tokenizer.apply_chat_template(msg["message"], tokenize=False, add_generation_prompt=True) for msg in batch]
        sampling_params = SamplingParams(n=1, temperature=0.6, max_tokens=32000, stop="<|eot_id|>", skip_special_tokens=True)

        # 开始计时（首次处理该 item 时）
        for msg in batch:
            idx = msg["item_index"]
            if idx not in item_start_flags:
                item_start_flags[idx] = time.time()

        # 模型生成
        batch_output = model.generate(prompts, sampling_params)

        for data_idx, output in enumerate(batch_output):
            for generated_idx, text_output in enumerate(output.outputs):
                generated_text = text_output.text
                answer_match = re.search(r"</think>\n(.*)", generated_text, re.DOTALL)
                if answer_match:
                    generated_text = answer_match.group(1)

                item_index = batch[data_idx]["item_index"]
                score = extract_score(generated_text)
                item_scores[item_index].append(score)

                # 若该 item 所有 constraint 都已处理完，记录时间
                expected_count = len(data[item_index].get("constraint", []))
                if len(item_scores[item_index]) == expected_count and item_index not in item_times:
                    elapsed = time.time() - item_start_flags[item_index]
                    item_times[item_index] = round(elapsed, 2)
                    print(f"[Item {item_index}] Finished {expected_count} constraints in {elapsed:.2f} seconds.")

    # 写出结果
    with open(output_file, 'w', encoding='utf-8') as file:
        for item_index, item in enumerate(data):
            scores = item_scores.get(item_index, [])
            avg_score = sum(scores) / len(scores) if scores else 0.0
            result = {
                **item,
                "score": round(avg_score, 4),
                "inference_time": item_times.get(item_index, 0.0),
                "constraint_scores": scores
            }
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

    # ========== 计算每group_size个item的时间和，并求所有组的平均值 ==========
    all_times = [item_times.get(i, 0.0) for i in range(len(data))]  # 按item_index顺序获取时间

    # 按每group_size个一组分组（确保与数据一致）
    group_sums = []
    for i in range(0, len(all_times), group_size):
        group = all_times[i:i+group_size]
        group_sum = sum(group)
        group_sums.append(group_sum)
        print(f"Group {i//group_size + 1}: Items {i}~{i+len(group)-1}, Total Time = {group_sum:.2f}s")

    # 计算所有组的平均值
    if group_sums:
        avg_group_time = sum(group_sums) / len(group_sums)
        print(f"\n>>> Average time per {group_size}-item group: {avg_group_time:.2f} seconds")
    else:
        avg_group_time = 0.0
        print("No groups to calculate average.")

    # 将平均值写入summary文件（确保格式一致）
    summary_file = output_file.replace(".jsonl", "_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Total items: {len(data)}\n")
        f.write(f"Number of {group_size}-item groups: {len(group_sums)}\n")
        f.write(f"Average time per {group_size}-item group: {avg_group_time:.2f} seconds\n")
        for idx, gsum in enumerate(group_sums):
            f.write(f"Group {idx+1}: {gsum:.2f}s\n")
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"✅ Summary已保存到: {summary_file}")

if __name__ == "__main__":
    main()

