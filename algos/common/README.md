# Common Class for RL Algorithms

## Actor

### Continuous Space Actor
- sample(self, state, deterministic) -> unnorm_action, norm_action:
    - can save internal variables here such as action distributions, latent variables for GRU, and etc.
- getDist(self, state):
    - If state is None, use internal state.
- getEntropy(self, state):
    - If state is None, use internal state.
- getLogProb(self, state, action):
    - If state is None, use internal state.
- initialize(self):
    - For network initialization.

---
## Model Config

