import json
from collections import defaultdict

# 软约束集合
soft_constraints = set([
    'soft_content:language', 'soft',
    'soft_content:open_ended',
    'situation:suggestion', 'situation:role_play',
    'situation:story_generation', 'style:open_ended'
])

# 初始化统计结构
item_count_by_len = defaultdict(int)
constraint_stats_by_len = {
    i: {"total": 0, "soft": 0, "hard": 0} for i in range(1, 6)
}

# 替换为你的文件名
file_path = "/oss/rqy/qingyu/verl/data/train_all_scaler_processed_2w_dp.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        constraints = item.get("instruction_id_list", [])
        length = len(constraints)

        # 只处理长度 1-5
        if not (1 <= length <= 5):
            continue

        # 跳过 ["light"] 或 ["light_choice"]
        if length == 1 and constraints[0] in ["light", "light_choice"]:
            continue

        # 统计符合条件的 item 数
        item_count_by_len[length] += 1

        # 统计软硬约束数量
        for c in constraints:
            constraint_stats_by_len[length]["total"] += 1
            if c in soft_constraints:
                constraint_stats_by_len[length]["soft"] += 1
            else:
                constraint_stats_by_len[length]["hard"] += 1

# 输出结果
print("【符合条件的 item 数量】")
for i in range(1, 6):
    print(f"长度为 {i} 的 item 数量: {item_count_by_len[i]}")

print("\n【每类中约束数量统计】")
for i in range(1, 6):
    stats = constraint_stats_by_len[i]
    print(f"长度为 {i}: 总约束 {stats['total']}，软约束 {stats['soft']}，硬约束 {stats['hard']}")
