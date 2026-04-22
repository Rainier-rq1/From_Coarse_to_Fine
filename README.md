# 🚀 From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks
[![Github](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=github&message=Rainier-rq1/From_Coarse_to_Fine)](https://github.com/Rainier-rq1/From_Coarse_to_Fine)
![image](https://raw.githubusercontent.com/Rainier-rq1/From_Coarse_to_Fine/main/WEval/q.png)

# Evaluation--WEval

- Enter the directory
    ```bash
   cd WEval
   ```

# Training--WRL
## 🔧 Environment Setup
- Enter the directory
    ```bash
   cd WRL/verl
   ```
- Python and required dependencies
  ```bash
   pip install -e .
   ```
---
## 1. Start RM Service
1. Open and edit `RM.py`:
   - Modify the **model_name** in the script to your trained reward model's path.
2. Run on development machine:
   ```bash
   python RM.py
   ```
3. After successful startup, the console will output the **IP address** and **port number** of the RM service.
---
## 2. Configure Reward Interface
1. Open `verl/utils/reward_score/instruction_reward.py`
2. Replace the **IP and Port** in the file with the address displayed when the RM service started.
---
## 3. RL Training 
1. Edit the training startup script:
   ```bash
   examples/qwen2_7b_instruction.sh
   ```
   - Replace the **model checkpoint path** with the actual path
2. Modify the configuration file:
   ```bash
   config.yaml
   ```
   - Configure the correct **model path**
   - Configure  other parameters
3. Start training (execute on master node):
   ```bash
   sh examples/qwen2_7b_instruction.sh
   ```
---
