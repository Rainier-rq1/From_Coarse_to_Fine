import json
import requests
import time
import os
import re

def infer_group_size_from_filename(filename):
    """从文件名推断group_size（3c, 4c, 5c）"""
    basename = os.path.basename(filename)
    # 匹配3c, 4c, 5c模式
    match = re.search(r'(\d+)c', basename)
    if match:
        return int(match.group(1))
    # 默认值
    return 3

input_file = 'eval.jsonl'
output_file = 'rm.jsonl'
url = "http://0.0.0.0:55111/predict"

# 自动推断group_size
group_size = infer_group_size_from_filename(input_file)
print(f"检测到输入文件: {os.path.basename(input_file)}")
print(f"自动设置 group_size = {group_size}")

# 用于累计当前组的耗时
current_batch = []

# 用于存储每组的总耗时
group_durations = []

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin, 1):
        item = json.loads(line.strip())
        constraints = item.get("constraint", [])
        answer = item.get("chosen", "")
        
        res = 0.0
        start_time = time.time()
        
        for question in constraints:
            data = {
                "answer": answer,
                "question": question
            }
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                result = response.json()
                res += result.get("value", 0.0)
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        if constraints:
            item["score"] = res / len(constraints)
        else:
            item["score"] = 0.0
        
        item["duration"] = duration
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 累加到当前组
        current_batch.append(duration)
        
        # 每group_size个item组成一组，计算并记录组总耗时
        if idx % group_size == 0:
            group_sum = sum(current_batch)
            group_durations.append(group_sum)
            print(f"Items {idx-group_size+1} ~ {idx} total duration: {group_sum:.3f} seconds")
            current_batch.clear()  # 清空当前组，准备下一组
        
        print(f"Item {idx} processed in {duration:.3f} seconds")

# 处理最后不足group_size个的组
if current_batch:
    group_sum = sum(current_batch)
    start_idx = len(group_durations) * group_size + 1
    end_idx = start_idx + len(current_batch) - 1
    group_durations.append(group_sum)
    print(f"Final incomplete group Items {start_idx} ~ {end_idx} total duration: {group_sum:.3f} seconds")

# 最后计算"组间平均耗时"
if group_durations:
    avg_group_duration = sum(group_durations) / len(group_durations)
    print(f"\n📊 共 {len(group_durations)} 组，组间平均耗时: {avg_group_duration:.3f} 秒")
else:
    print("⚠️ 无完整组数据。")

print(f"\n✅ 结果已保存到: {output_file}")

