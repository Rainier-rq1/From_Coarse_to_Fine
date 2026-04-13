import pandas as pd

# 读取 JSONL 文件，注意每行是一个独立的 JSON 对象
df = pd.read_json('/oss/rqy/qingyu/verl/data/writing.jsonl', orient='records', lines=True)

# 将 DataFrame 转换为 Parquet 文件
df.to_parquet('/oss/rqy/qingyu/verl/data/writing.parquet', engine='pyarrow')  # 或使用 'fastparquet' 作为引擎
