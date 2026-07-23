
# Reproduction of RL-Algorithms

--------------------------------------
Reproduction of RL papers that I found interesting (mainly for learning purposes). I try to focus on reproducibility and getting the scores in the original papers. Moreover, I also try to stress-test, explore and extend (if I think of something) the papers. 


I also have some other implementations that I made before I decided to centralize everything under one repo, they can be found here:
- [DQN-DRQN](https://github.com/aryangarg794/DQN-DRQN)
- [Dueling DQN](https://github.com/aryangarg794/Dueling-DQN)
- [TD3](https://github.com/aryangarg794/TD3)
- [Q-Learning/SARSA/E-SARSA](https://github.com/aryangarg794?tab=repositories)
- [DPG](https://github.com/aryangarg794/DPG)
- [DDPG](https://github.com/aryangarg794/DDPG)
- [REINFORCE/A2C](https://github.com/aryangarg794/Policy-Methods)


----------


## Setup

This project utilizes `uv` for fast, reproducible dependency management and virtual environment isolation. Ensure you have `uv` installed on your system.

### Installing uv

On windows you can download uv from their download page. 
```bash
# for macOS / Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh 

uv sync

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Basic Run Guide for Typer

I'm using `typer` for my CLI package. To check the algorithms implemented you can run:

```bash
uv run python src/rl_algorithms/main.py --help
```

The algorithms are under commands:

```bash
Usage: main.py [OPTIONS] COMMAND [ARGS]...                                                                                                                      
                                                                                                                                                                 
 RL Algorithms main.py                                                                                                                                           
                                                                                                                                                                 
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                       │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                │
│ --help                        Show this message and exit.                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ trpo                                                                                                                                                          │
│ ppo                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

Each algorithm has their own arguments which can be checked using `--help`. Then we can run the algorithm like this:

```bash
uv run python src/rl_algorithms/main.py trpo [ARGS]
```

Example `--help` for TRPO:

```bash
 Usage: main.py [OPTIONS]                                                                                                                                        
                                                                                                                                                                 
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --env-id                                 <str>    environment to test on (has to be gym env) [required]                                                    │
│    --lr-critic                              <float>  lr for critic [default: 0.001]                                                                           │
│    --discount                               <float>  gamma [default: 0.99]                                                                                    │
│    --trust-region                           <float>  trust region coeff [default: 0.01]                                                                       │
│    --act-hidden                             <int>    actor hidden dims [default: 400, 300]                                                                    │
│    --critic-hidden                          <int>    critic hidden dims [default: 400, 300]                                                                   │
│    --damping                                <float>  damping coeff [default: 0.0]                                                                             │
│    --device                                 <str>    device [default: cuda]                                                                                   │
│    --num-critic-updates                     <int>    num critic updates per actor updates [default: 10]                                                       │
│    --norm-adv              --no-norm-adv             normalize adv [default: norm-adv]                                                                        │
│    --n-envs                                 <int>    num of envs [default: 5]                                                                                 │
│    --batch-size                             <int>    batch size [default: 128]                                                                                │
│    --n-steps                                <int>    num of steps per rollout [default: 512]                                                                  │
│    --total-timesteps                        <int>    num total training steps [default: 100000]                                                               │
│    --window-size                            <int>    window size to average the ep rews [default: 100]                                                        │
│    --seeds                                  <int>    seeds to run [default: 0, 1, 2, 3]                                                                       │
│    --save                  --no-save                 save model/results [default: no-save]                                                                    │
│    --run-name                               <str>    name of the run [default: trpo]                                                                          │
│    --install-completion                              Install completion for the current shell.                                                                │
│    --show-completion                                 Show completion for the current shell, to copy it or customize the installation.                         │
│    --help                                            Show this message and exit.                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

----------
## Results

The results of each algorithm reproduction/investigation can be found here
- [TRPO Results](results/trpo/trpo.md)