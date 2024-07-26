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

## 2024-07-19

### run different Lipschitz gradient (AHU2)
In this case, the DGD seems better than the BB and the BCGD.

### Change the strategy of Byzantine agents to make the attacks more fierce.

The strategy of the Byzantine agent is added to the algorithm_setting in the config file with 
```yaml
algorithm:
  alg_settings:
    byz_msg:
      byz_plan: 'rand'  # rand, flipping, times, ones
      byz_mag: 9    # 
```
- rand: generate random number
- flip: -param to transmite
- time: magnitude*param
- ones: byz_mag*ones

And the configuration files are named _planrand9_ something like this.


The utility is updated to check the config and the name of config file.

From the results, it seems that the result in `time9` diverge quickly 
(possibly because of the initialization is too far from opt) 

For other cases, the dgd with diminishing stepsize shows best performance at the initial, 
but quickly meet the performance of dgd with constant stepsize. 

`rand9` seems most suitable for the clarification.

### Next time change the AHU to smaller value 1.1 and mild initialization as well as shorten end (400)
AHU to smaller value 1.1, such that the weight of bcgd will be more relaxed.

The configs are used as AHU11e1. 

__Because different numbers of Byzantine agents shows similar performance, byz is set to be 2.__

[16H31M18S_run_algorithms_ss1e2_byz2_init10_AHU11e1_planxxxx9.sh](16H31M18S_run_algorithms_ss1e2_byz2_init10_AHU11e1_planxxxx9.sh)
This file try different attacks and see the results to campare with different attacks 
and see what attakcs is most attractive for comparisons.

### change the magnitude of attack and see the results.

# V1.1 start to use same problems and networks.
- Change the initialization of RayAgent.py
  - Instead of generate the problem on each agent, we first generate the problem and then share them to agents.
  - check where `RayAgent` is initialized.
    - 因为是`ray.remote()`所以没法直接找到。还是通过搜索的方式找`RayAgent.options(name=f'{agent_id}'` 看看能搜到什么不能。
      - They appear in the `__init__` of the algorithms.
      - Change from `prob_settings=self.args.prob_settings,` to `problem=self.common_problems[agent_id],` 
    - 从`load_data` branch的代码中吸取经验，通过新的main函数进行书写。
- `LeastSquareProblem` class add $M$ and $m$ to its attribution.
  - `self.M`, `self.m`
  - `self.b_cond` for the condition of b.
- New `data` folder with `shared_networks` and `shared_problems` as subfolders.
- Restructured the `least_square_problem.py`.
- Add `network_io.py` for the load, save of common network, they are moved from utils (in branch load_data)
  - save_common_network
  - load_common_network
- Changed the `prob_info` to `prob_settings` from the yaml config file to the codes.
- Change the config files' setting:
  - algorithm to algorithms (list when read), see `test_207_07H14M29S_new.yml` for details.
    - see `alg_name` in `algorithm` 
    - change the default config in`utils.py`.

- Changed the algorithm related initialization contents.

  ```python
  def __init__(self, args, common_network: Network, common_problems: AbstractProblem) -> None:
      self.args = args
      # self.network = Network(net_settings=args.net_settings)
      self.network = common_network
      print(self.network.agent_types)
  
      self.common_problems = common_problems
  ```

# V1.3 Change the name of logs



- Move the `centralized_solver` from `abstract-algorithm` to `utils`

  ```python
  def centralized_solver(self) -> np.ndarray:
      """Compute the global centralized solution."""
      prob_futures = [agent.get_prob.remote() for agent in self.agents.values()
                      if ray.get(agent.get_type.remote()) == 'Reg']
      setup_timeout = self.timeout_base*self.network.net_size
      ready_probs, _ = self.wait_for_all_futures(prob_futures, setup_timeout, "get_prob")
  
      probs = [ray.get(ready_prob) for ready_prob in ready_probs]
      self.prob_a_list = [prob.A for prob in probs]
      sum_a = np.sum(self.prob_a_list, axis=0)
      self.prob_b_list = [prob.b for prob in probs]
      sum_b = np.sum(self.prob_b_list, axis=0)
      if self.verbose:
          print(f"A for agents are: {self.prob_a_list}")
          print(f"b for agents are {self.prob_b_list}")
      return QuadraticFunction(sum_a, sum_b).closed_form_minimizer()
      # return (-np.linalg.inv(0.5 * (sum_a.T + sum_a)) @ sum_b).flatten()
  ```

- removed `prob_a_list` and `prob_b_list` in `save_all_hist`, and add `config_filename` to logs.

- Change `get_all_hist` in the `abstract_algorithm.py` to parallel get.

- remove the redundant settings in config.yml.

- Move `save_config` to the `abstract_algorithm`.

- `generate_log_path` is changed form `generate_logging_path`. This helps to generate the path (filename) of the algorithm results and the configurations.

- add `kill` to main.py such that the agents can be free after the algorithm is finished. (removed and change init shutdown for ray.)

- Changed the `_load_yaml`, such that the configs are set as default to the folder `configs`, we only need to provide the. Now you can run it by `python main.py -a test_207_07H14M29S_new.yml`.

- loss function is added to RayAgent. Previous because of the delete of _setup_problems, this is not setup and therefore, the results are uncorrect. Change it here.

***

# V2.0 block_byz_aggregation

Previous codes are wrong. Because the `block_byz_aggregation` in RayAgent is uncorrect!!!.

## Changes before V2.1 

- Changed the weight in ray_agent 257 to avoid the case when there is only agent itself:

  > ```python
  > w_others = (1-wd) / (len(sel_ids)-1) if len(sel_ids) > 1 else 1.
  > ```

- Add FCGD (full consensus gradient descent) to the configs when blksiz = 1.
- changed the seed in LeastSquareProblem with agent_id added. Such that each agent can generate its own problem (previously, they generate the same problem, and the performance are same for different settings. That can be understood.)
- 
