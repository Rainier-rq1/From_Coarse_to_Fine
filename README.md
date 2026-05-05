# 🚀 From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks

[![Github](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=github&message=Rainier-rq1/From_Coarse_to_Fine)](https://github.com/Rainier-rq1/From_Coarse_to_Fine)
![image](https://raw.githubusercontent.com/Rainier-rq1/From_Coarse_to_Fine/main/WEval/q.png)

## 📢 Latest News

- **2026-04**: Our paper "From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks" has been accepted by **ACL 2026 Findings** ! 


---


## Evaluation: WEval

### Environment Setup

```bash
cd WEval
```

### Running Inference

#### 1. Generation Reward Model

```bash
python WritingBench-Critic.py
python llm_judge.py
```

#### 2. Scalar Reward Model

```bash
python server.py
python server_eval.py
```

### Computing Correlation

```bash
python cal_corr.py
```

---

## Training: WRL

### Environment Setup

```bash
cd WRL/verl
pip install -e .
```

### Step 1: Start the Reward Model Service

1. Open and edit `RM.py`:
   - Set the **model_name** to your trained reward model's path.
2. Launch the service:
   ```bash
   python RM.py
   ```
3. Note the **IP address** and **port number** displayed in the console.

### Step 2: Configure Reward Interface

1. Open `verl/utils/reward_score/instruction_reward.py`.
2. Replace the IP and Port with the values from Step 1.

### Step 3: Start RL Training

1. Edit the training script `examples/qwen2_7b_instruction.sh`:
   - Update the **model checkpoint path** to your actual path.
2. Configure `config.yaml`:
   - Set the correct **file paths**.
   - Adjust other parameters as needed.
3. Launch training:
   ```bash
   sh examples/qwen2_7b_instruction.sh
   ```

---

