# Selective Superadiance into a Desired Channel

# PhD Project: Advanced Computational Methods for Atom Dynamics

In this project, I illustrate an example project of my PhD, showcasing how I use advanced computational methods such as higher order mean field methods and parallel computing to propagate the dynamics of a system of many atoms (N >~ 100) using minimal compute resources (a computer with 8 cores and 4 GB of RAM). If more compute resources are available, such as on a computing cluster, the provided code runs in reasonable time for (N ~ 1000).

## Demonstration File

The demonstration file is [selective_superadiance.ipynb](selective_superadiance.ipynb). Go through this file to learn more about the idea of selective superadiance and the steps used to build the simulation.

## Code Structure

I write my code in a modular way:

- **[selective_superadiance.ipynb](selective_superadiance.ipynb)**: The main file illustrating an example workflow in my PhD.
    - Workflow sequence:
        1. Define the system parameters
        2. Visualize the system
        3. Run the main evolution routine
        4. Gather and plot the data
- **[atom_recoil_func.py](atom_recoil_func.py)**: Contains functions that define the freespace Green function matrix, controlling the dissipative and coherent coupling between the atoms depending on their locations and the polarization of the transition.
- **[mf2_func.py](mf2_func.py)**: Contains the main evolution subroutine, which includes the dynamical system differential equations for a second order mean field (MF2) approximation.
- **[save_data.py](save_data.py)**: Used to save the time series from the simulation into a tabulated .csv file for later plotting. It also logs the time, parameters, source code, and output files for each simulation for future reference.

## Contact

If you have any questions, reach out to me through [my email](mailto:tohfa@aucegypt.edu).

## Requirement

Ensure that the libraries listed at the header of each file are installed if they are not already available. In particular, I used **Numba** to accelerate the performance of numerical computations in Python. Numba is a Just-in-Time (JIT) compiler that translates a subset of Python and NumPy code into fast machine code. It is particularly useful for speeding up loops and array operations, making it an excellent choice for computational tasks in scientific computing.

To install Numba, you can use pip:

```bash
pip install numba
```

Make sure to check the compatibility of Numba with your Python version and other libraries you are using.
