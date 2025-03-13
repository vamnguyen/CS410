import numpy as np
import matplotlib.pyplot as plt
from src.objective_functions import get_function
from src.de import DifferentialEvolution
from src.cem import CrossEntropyMethod

def run_example():
    # Set parameters
    func_name = 'sphere'
    dim = 2
    pop_size = 32
    seed = 22520880  # MSSV
    max_evals = 2000
    
    # Create objective function
    obj_func = get_function(func_name, dim)
    
    # Run DE
    print("Running DE...")
    de = DifferentialEvolution(obj_func, pop_size, max_evals=max_evals)
    de_best_solution, de_best_fitness = de.optimize(seed=seed)
    
    # Reset function evaluations counter
    obj_func.reset_evaluations()
    
    # Run CEM
    print("Running CEM...")
    cem = CrossEntropyMethod(obj_func, pop_size, max_evals=max_evals)
    cem_best_solution, cem_best_fitness = cem.optimize(seed=seed)
    
    # Print results
    print("\nResults:")
    print(f"DE best solution: {de_best_solution}")
    print(f"DE best fitness: {de_best_fitness}")
    print(f"CEM best solution: {cem_best_solution}")
    print(f"CEM best fitness: {cem_best_fitness}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(de.num_evaluations, de.best_fitnesses, 'b-', label='DE')
    plt.plot(cem.num_evaluations, cem.best_fitnesses, 'r-', label='CEM')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value')
    plt.title(f'{func_name.capitalize()} (d={dim}, N={pop_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig('example_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_example() 