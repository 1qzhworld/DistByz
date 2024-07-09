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



