# Safe RL Algorithms

## Prerequisites

```bash
gymnasium==0.28.1
safety-gymnasium==1.0.0
torch==2.0.1
qpsolvers==1.9.0
scipy==1.10.1
```

## How to run

`python main_saferl.py --task_cfg_path tasks/point_goal.yaml --algo_cfg_path algos/safe_rl/lppo/safetygym.yaml --wandb`