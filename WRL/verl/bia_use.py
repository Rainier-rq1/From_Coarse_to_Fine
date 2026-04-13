from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
model_name = "/mnt/shared-storage-user/renqingyu/models/models--renqingyu--renqingyu-rm_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    torch_dtype="auto",
).to("cuda")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({'error': 'Missing question or answer in request'}), 400
        
        question = data['question']
        answer = data['answer']
        
        # Format input text
        text = f"You need to determine whether the response adheres to this constraint. Only output  1 if it does, otherwise only output  0.  Response: {answer}. Constraint: {question}"
        # text = f"""
        #     You need to determine whether a response satisfies the given constraint. The type of constraint relationship must be automatically inferred from the context.

        #     There are five possible types:

        #         1. Single: The response must satisfy one constraint.

        #         2. And : The response must satisfy all listed constraints simultaneously.

        #         3. Chain : The response must complete multiple tasks sequentially, in order.

        #         4. Selection : The response must follow branching conditions and satisfy the correct branch.

        #         5.Nested : The response involves nested combinations of And/Chain/Selection.

        #     Finally, you must output: 1 if the response satisfies the constraint; 0 if it does not.

        #     Response: {answer} Constraint: {question}"""
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probas = F.softmax(logits, dim=-1)
        value = round(probas[0][1].item(), 3)
        pred = logits.argmax(dim=-1).item()
        return jsonify({
            'value': value,
            "pred" : pred,
            'question': question,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=55111)