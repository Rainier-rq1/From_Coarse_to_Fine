from vllm import LLM, SamplingParams
import os
import json
import re
import time
from collections import defaultdict
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


evaluate_prompt = """
Evaluate the Response based on the  Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

-Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.

-Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.

-Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.

-Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.

-Scoring Range: Assign an integer score strictly between 1 to 10

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score strictly between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}

** Criteria **
```{criteria}```


** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
""".strip()

def extract_score(text):
    print(text)
    match = re.search(r'"score"\s*:\s*(\d+)', text)
    if match:
        try:
            return int(match.group(1))
        except:
            return 0
    return 0

def infer_group_size_from_filename(filename):
    """从文件名推断group_size（3c, 4c, 5c）"""
    basename = os.path.basename(filename)
    # 匹配3c, 4c, 5c模式
    match = re.search(r'(\d+)c', basename)
    if match:
        return int(match.group(1))
    # 默认值
    return 3

def process_question(item):
    response = item.get('chosen')
    constraints = item.get('constraint', [])
    query = item.get('prompt')

    prompts = []
    for checker in constraints:
        prompt = evaluate_prompt.format(
            criteria=checker,
            query=query,
            response=response
        )    
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

def main():
    model_path = "models--AQuarterMile-WritingBench-Critic-Model-Qwen-7B"
    model = LLM(model_path, dtype='bfloat16', gpu_memory_utilization=0.8, tensor_parallel_size=4, max_num_seqs=256)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_file = "eval.jsonl"
    output_file = "wb.jsonl"

    # 自动推断group_size
    group_size = infer_group_size_from_filename(input_file)
    print(f"检测到输入文件: {os.path.basename(input_file)}")
    print(f"自动设置 group_size = {group_size}")
    print("-" * 60)

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

    # 按每group_size个一组分组
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

    # 可选：将平均值写入一个 summary 文件
    summary_file = output_file.replace(".jsonl", "_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Total items: {len(data)}\n")
        f.write(f"Number of {group_size}-item groups: {len(group_sums)}\n")
        f.write(f"Average time per {group_size}-item group: {avg_group_time:.2f} seconds\n")
        for idx, gsum in enumerate(group_sums):
            f.write(f"Group {idx+1}: {gsum:.2f}s\n")

if __name__ == "__main__":
    main()
