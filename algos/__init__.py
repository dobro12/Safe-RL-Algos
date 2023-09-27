from algos.rl.trpo import Agent as TRPO
from algos.rl.ppo import Agent as PPO

# from algos.safe_rl.offtrc import Agent as OffTRC
# from algos.safe_rl.wcsac import Agent as WCSAC
from algos.safe_rl.lppo import Agent as LPPO
from algos.safe_rl.ipo import Agent as IPO

algo_dict = {
    'trpo': TRPO,
    'ppo': PPO,
    # 'offtrc': OffTRC,
    # 'wcsac': WCSAC,
    'lppo': LPPO,
    'ipo': IPO,
}