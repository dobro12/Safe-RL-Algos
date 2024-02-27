from safe_rl_algos.algos.rl.trpo import Agent as TRPO
from safe_rl_algos.algos.rl.ppo import Agent as PPO

# from safe_rl_algos.algos.safe_rl.offtrc import Agent as OffTRC
# from safe_rl_algos.algos.safe_rl.wcsac import Agent as WCSAC
from safe_rl_algos.algos.safe_rl.lppo import Agent as LPPO
from safe_rl_algos.algos.safe_rl.ipo import Agent as IPO
from safe_rl_algos.algos.safe_rl.p3o import Agent as P3O
from safe_rl_algos.algos.safe_rl.p3o_asym import Agent as P3O_ASYM

algo_dict = {
    'trpo': TRPO,
    'ppo': PPO,
    # 'offtrc': OffTRC,
    # 'wcsac': WCSAC,
    'lppo': LPPO,
    'ipo': IPO,
    'p3o': P3O,
    'p3o_asym': P3O_ASYM,
}