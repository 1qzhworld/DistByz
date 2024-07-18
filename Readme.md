## 2024-07-03
The `functions`, `networks`, and `problems` are copied from BlockBYZ__Personal2024-06-20
The `networks` requires no change, but the algorithm is not run in parallel. Still need improvements.
Today, I removed the communicator class and implement the algorithm and communication in the ray agent class.

### Something about communicators
`communicators/communicator.py` and `mpi_communicator.py` are based on the disropt.
However, as the code is hard to debug. These codes are not used for the communicator.
In this project, the ray_communicator is essentially used as a communicator of agent.

## 2024-07-09

This time it works but there are too much worker therein. I am trying to change the structure of this by refactor the functions and classes.

__copied from v0__



## 2024-07-13
### One future direction:
Make the block selection for agents different.
Currrently they are transmite through the algorithm, but not initialized in each agent. 

currently the blk_size and related blk_indexes are calculate in the algorithm instead of the agent. Because in the previous setting it implies that each agent can have its own block split methods and the blocks can be different in the agents. Therefore, is different from the algorithm in the analysis. We have changed this to make the algorithm more robust by ensure the blocks are same for all agents.


## 2024-07-15
Currently, there is only a few things remain to be done. 
For example, to record the distance to optimal and plot the results.

## 2024-07-18
Change the stepsize to 1e-2, because the theoretical result on the stepsize is from $(..., 1/L]$. 
When $L=10$, we can set stepsize to 1e-2.

### Larger blocksize leads to better convergence.

### 2024-07-18 14H:03M:41S: change the initialization points to 100*np.ones(dimension)