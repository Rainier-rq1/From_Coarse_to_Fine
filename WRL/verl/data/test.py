def split_jsonl(input_file, output_file, start_line=20001):
    """
    从 input_file 的 start_line 行开始，将剩余内容写入 output_file
    :param input_file: 原始 JSONL 文件路径
    :param output_file: 新 JSONL 文件路径
    :param start_line: 开始行数 (从 1 开始计数)
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, start=1):
            if i >= start_line:
                outfile.write(line)

# 使用示例
input_path = "/oss/rqy/qingyu/verl/data/writing.jsonl"
output_path = "/oss/rqy/qingyu/verl/data/writing_subset.jsonl"
split_jsonl(input_path, output_path, start_line=20001)
