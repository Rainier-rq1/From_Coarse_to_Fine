import json
from scipy.stats import kendalltau

def position_consistency(rank1, rank2):
    diffs = [abs(a - b) for a, b in zip(rank1, rank2)]
    avg_diff = sum(diffs) / len(diffs)
    max_diff = len(rank1) - 1
    consistency = 1 - (avg_diff / max_diff)
    return consistency

def process_jsonl(input_path, output_path, group_size=4):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    updated_data = []
    kendall_tau_values = []
    position_consistency_values = []
    exact_match_group_values = []  # 每组是否完全一致

    for i in range(0, len(data), group_size):
        group = data[i:i + group_size]
        if len(group) < group_size:
            break

        sorted_by_score = sorted(
            group,
            key=lambda x: (x['score'], len(x.get('instruction_id_list', []))),
            reverse=True
        )
        score_rank_map = {id(item): rank for rank, item in enumerate(sorted_by_score, 1)}

        for item in group:
            item['label'] = group_size - len(item.get('instruction_id_list', []))

        sorted_by_label = sorted(group, key=lambda x: x['label'], reverse=True)
        label_rank_map = {id(item): rank for rank, item in enumerate(sorted_by_label, 1)}

        ranking_seq = [score_rank_map[id(item)] for item in group]
        label_seq = [group_size + 1 - label_rank_map[id(item)] for item in group]

        for item in group:
            item['ranking'] = score_rank_map[id(item)]

        tau, _ = kendalltau(ranking_seq, label_seq)
        kendall_tau_values.append(tau)

        pos_consistency = position_consistency(ranking_seq, label_seq)
        position_consistency_values.append(pos_consistency)

        # === 新增：判断这一组是否完全一致 ===
        exact_match_group = 1 if ranking_seq == label_seq else 0
        exact_match_group_values.append(exact_match_group)

        print(f"Group {i//group_size + 1} ranking sequence:      {ranking_seq}")
        print(f"Group {i//group_size + 1} label sequence:        {label_seq}")
        print(f"Group {i//group_size + 1} Kendall Tau coefficient: {tau:.4f}")
        print(f"Group {i//group_size + 1} Position Consistency:   {pos_consistency:.4f}")
        print(f"Group {i//group_size + 1} Exact Match (group):    {exact_match_group}")
        print('---')

        updated_data.extend(group)

    avg_tau = sum(kendall_tau_values) / len(kendall_tau_values) if kendall_tau_values else 0
    avg_pos_consistency = sum(position_consistency_values) / len(position_consistency_values) if position_consistency_values else 0
    avg_exact_match_group = sum(exact_match_group_values) / len(exact_match_group_values) if exact_match_group_values else 0

    print(f"\nAverage Kendall Tau coefficient across all groups: {avg_tau:.4f}")
    print(f"Average Position Consistency across all groups:   {avg_pos_consistency:.4f}")
    print(f"Average Exact Match across all groups:            {avg_exact_match_group:.4f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 示例调用
input_file = 'qwen.jsonl'
output_file = 'output_ranked.jsonl'
process_jsonl(input_file, output_file, group_size=5)

