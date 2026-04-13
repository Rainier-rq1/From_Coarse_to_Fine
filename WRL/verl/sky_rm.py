


#!/usr/bin/env python3
"""
Skywork-Reward-V2-Qwen3-8B Flask 服务器
使用官方示例的处理逻辑
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
model_name = "/mnt/shared-storage-user/renqingyu/models/Skywork-Reward-V2-Qwen3-8B"

print(f"正在加载模型: {model_name}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

# 构建加载参数
load_kwargs = {
    "num_labels": 1,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    "trust_remote_code": True
}

# 如果支持 flash_attention_2，添加该参数
if torch.cuda.is_available():
    try:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    except:
        pass  # 如果不支持，忽略

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    **load_kwargs
).to(device)

model.eval()
print(f"模型加载完成，设备: {device}")

@app.route('/predict', methods=['POST'])
def predict():
    """使用官方示例的处理逻辑"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({'error': 'Missing question or answer in request'}), 400
        
        question = data['question']
        answer = data['answer']
        
        # 使用官方示例的格式：构建对话格式
        conv = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # 使用 apply_chat_template 格式化对话（官方方式）
        conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        
        # 移除可能的重复 bos token（官方方式）
        if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token):]
        
        # Tokenize
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
        
        # Get the reward score（官方方式：直接取 logits[0][0]）
        with torch.no_grad():
            score = model(**conv_tokenized).logits[0][0].item()
        
        # 对于 num_labels=1，不需要 softmax 和 argmax
        # 直接使用 logits 值作为奖励分数
        value = round(score, 3)
        
        # pred 可以根据分数阈值判断（例如 > 0 为 1，否则为 0）
        pred = 1 if value > 0 else 0
        
        return jsonify({
            'value': value,
            'pred': pred,
            'question': question,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=55111)

