# Lyapunov-Actor-Critic
My implementation of the Actor Critic algorithm with stability guarantee following this paper: https://arxiv.org/pdf/2004.14288

Author's Implementation: https://github.com/hithmh/Actor-critic-with-stability-guarantee

** Note: I believe the paper has a typo in there appendix and the hyperparameter alpha3 should be 0.1 not 1 as noted in the paper. I cloned the source code that the authors link to in the paper and noted that all the experiment can be recreated with an alpha3 of 0.1 but not 1.

Direct comparisons are not made as my enviornments do differ slightly. I use MUJOCO to create the cart pole environment. This is not necessary and one can write the dynamics themselves quite easily but I also wanted to learn more about custom MUJOCO environments as well. 
