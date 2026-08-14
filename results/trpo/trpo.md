# TRPO Reproduction & Exploration

----------


### Reproduction results

Running on several environments from Gymnasium shows that the model is able to learn and solve certain environments. 


Notes 

- Difficult to optimize the certain parameters
- intialization of conugate gradients does not matter
- it trained much better from the async rollout with different envs vs when I had a single env
- damping may be important
- initialization is really important, cartpole doesnt reach 500 wihtout proper init, but inverted pendulum sometimes breaks with init 