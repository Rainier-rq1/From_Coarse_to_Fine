import matplotlib.pyplot as plt
import numpy as np

# 三组数据
data1 = {
    'Math': 4501,
    'Science': 1929,
    'Instruction\nfollowing': 13570
}

data2 = {
    'Soft\nconstraints': 24196,
    'Hard\nconstraints': 16095
}

data3 = {
    '1 constraint': 2806,
    '2 constraints': 2745,
    '3 constraints': 2700,
    '4 constraints': 2700,
    '5 constraints': 2619
}

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

all_data = [data1, data2, data3]
titles = ['Task Type Count', 'Constraint Type Count', 'Constraint Number Distribution']

for i, ax in enumerate(axes):
    data = all_data[i]
    labels = list(data.keys())
    values = list(data.values())
    x = np.arange(len(labels))
    
    # 绘图
    bars = ax.bar(x, values, color='#88C999')
    
    # 设置标题和x轴标签
    ax.set_title(titles[i], fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, rotation=15)
    ax.tick_params(axis='y', labelsize=12)

    # 设置更宽松的y轴上限
    max_val = max(values)
    margin = max(0.1 * max_val, 500)  # 至少留500的空间
    ax.set_ylim(0, max_val + margin)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + margin * 0.05, f'{height}', 
                ha='center', va='bottom', fontsize=11)

# 布局优化
plt.tight_layout()
plt.savefig('three_bar_charts_better_margin.pdf', format='pdf')
plt.show()
