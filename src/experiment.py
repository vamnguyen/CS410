import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.objective_functions import get_function
from src.de import DifferentialEvolution
from src.cem import CrossEntropyMethod
from src.visualization import plot_convergence, create_contour_animation, create_results_table, format_table_for_latex

def run_single_experiment(func_name, dim, pop_size, seed, max_evals=None):
    """
    Run a single experiment with both DE and CEM
    
    Parameters:
    -----------
    func_name : str
        Name of the objective function
    dim : int
        Dimension of the problem
    pop_size : int
        Population size
    seed : int
        Random seed
    max_evals : int, optional
        Maximum number of function evaluations
    
    Returns:
    --------
    results : dict
        Dictionary containing results for both algorithms
    """
    # Create objective function
    obj_func = get_function(func_name, dim)
    
    # Set max evaluations if not provided
    if max_evals is None:
        max_evals = 2000 if dim == 2 else 10000
    
    # Run DE
    de = DifferentialEvolution(obj_func, pop_size, max_evals=max_evals)
    de_best_solution, de_best_fitness = de.optimize(seed=seed, verbose=False)
    
    # Reset function evaluations counter
    obj_func.reset_evaluations()
    
    # Run CEM
    cem = CrossEntropyMethod(obj_func, pop_size, max_evals=max_evals)
    cem_best_solution, cem_best_fitness = cem.optimize(seed=seed, verbose=False)
    
    # Collect results
    results = {
        'func_name': func_name,
        'dim': dim,
        'pop_size': pop_size,
        'seed': seed,
        'max_evals': max_evals,
        'de': {
            'best_solution': de_best_solution.tolist(),
            'best_fitness': float(de_best_fitness),
            'best_solutions': [s.tolist() for s in de.best_solutions],
            'best_fitnesses': de.best_fitnesses,
            'num_evaluations': de.num_evaluations,
            'population_history': [p.tolist() for p in de.population_history] if dim == 2 else None
        },
        'cem': {
            'best_solution': cem_best_solution.tolist(),
            'best_fitness': float(cem_best_fitness),
            'best_solutions': [s.tolist() for s in cem.best_solutions],
            'best_fitnesses': cem.best_fitnesses,
            'num_evaluations': cem.num_evaluations,
            'population_history': [p.tolist() for p in cem.population_history] if dim == 2 else None
        }
    }
    
    return results

def run_experiments(mssv, func_names, dims, pop_sizes, num_runs=10):
    """
    Run experiments for all combinations of parameters
    
    Parameters:
    -----------
    mssv : int
        Student ID to use as base for random seeds
    func_names : list of str
        Names of objective functions to test
    dims : list of int
        Dimensions to test
    pop_sizes : list of int
        Population sizes to test
    num_runs : int
        Number of runs for each combination
    
    Returns:
    --------
    all_results : dict
        Dictionary containing all results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize results dictionary
    all_results = {}
    
    # Total number of experiments
    total_experiments = len(func_names) * len(dims) * len(pop_sizes) * num_runs
    
    # Run experiments
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for func_name in func_names:
            all_results[func_name] = {}
            
            for dim in dims:
                all_results[func_name][dim] = {}
                max_evals = 2000 if dim == 2 else 10000
                
                for pop_size in pop_sizes:
                    all_results[func_name][dim][pop_size] = {
                        'de': [],
                        'cem': []
                    }
                    
                    for run in range(num_runs):
                        seed = mssv + run
                        
                        # Run experiment
                        results = run_single_experiment(func_name, dim, pop_size, seed, max_evals)
                        
                        # Save results
                        all_results[func_name][dim][pop_size]['de'].append(results['de'])
                        all_results[func_name][dim][pop_size]['cem'].append(results['cem'])
                        
                        # Save log file
                        log_file = f"logs/{func_name}_d{dim}_N{pop_size}_seed{seed}.json"
                        with open(log_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        pbar.update(1)
    
    # Save all results
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def create_convergence_plots(all_results, pop_size=32):
    """
    Create convergence plots for all functions and dimensions
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all results
    pop_size : int
        Population size to use for plots
    """
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    for func_name in all_results:
        for dim in all_results[func_name]:
            # Extract results for the specified population size
            de_results = all_results[func_name][dim][pop_size]['de']
            cem_results = all_results[func_name][dim][pop_size]['cem']
            
            # Create plot
            save_path = f"figures/{func_name}_d{dim}_N{pop_size}_convergence.png"
            plot_convergence(de_results, cem_results, func_name, dim, pop_size, save_path)

def create_animations(all_results, mssv):
    """
    Create animations for 2D problems
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all results
    mssv : int
        Student ID used as seed
    """
    # Create animations directory if it doesn't exist
    os.makedirs('figures/animations', exist_ok=True)
    
    for func_name in all_results:
        if 2 in all_results[func_name]:
            # Get results for 2D problems with N=32 and seed=mssv
            pop_size = 32
            seed_idx = 0  # First run uses seed=mssv
            
            de_result = all_results[func_name][2][pop_size]['de'][seed_idx]
            cem_result = all_results[func_name][2][pop_size]['cem'][seed_idx]
            
            # Create objective function
            obj_func = get_function(func_name, 2)
            
            # Create animations
            de_save_path = f"figures/animations/{func_name}_d2_N{pop_size}_seed{mssv}_DE.gif"
            cem_save_path = f"figures/animations/{func_name}_d2_N{pop_size}_seed{mssv}_CEM.gif"
            
            # Convert population history from list to numpy array
            de_history = [np.array(p) for p in de_result['population_history']]
            de_best_solutions = [np.array(s) for s in de_result['best_solutions']]
            
            cem_history = [np.array(p) for p in cem_result['population_history']]
            cem_best_solutions = [np.array(s) for s in cem_result['best_solutions']]
            
            # Create animations
            create_contour_animation('DE', obj_func, de_history, de_best_solutions, mssv, de_save_path)
            create_contour_animation('CEM', obj_func, cem_history, cem_best_solutions, mssv, cem_save_path)

def create_results_tables(all_results):
    """
    Create results tables for all functions and dimensions
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all results
    
    Returns:
    --------
    tables : dict
        Dictionary containing tables for all functions and dimensions
    """
    tables = {}
    
    for func_name in all_results:
        tables[func_name] = {}
        
        for dim in all_results[func_name]:
            # Create table
            table = create_results_table(all_results[func_name][dim], func_name, dim)
            tables[func_name][dim] = table
            
            # Format table for LaTeX
            latex_table = format_table_for_latex(table, func_name, dim)
            
            # Save table
            table.to_csv(f"results/{func_name}_d{dim}_table.csv")
            latex_table.to_latex(f"results/{func_name}_d{dim}_table.tex", escape=False)
    
    return tables

def run_full_experiment(mssv):
    """
    Run the full experiment
    
    Parameters:
    -----------
    mssv : int
        Student ID to use as base for random seeds
    """
    # Define parameters
    func_names = ['sphere', 'griewank', 'rosenbrock', 'rastrigin', 'ackley']
    dims = [2, 10]
    pop_sizes = [8, 16, 32, 64, 128]
    
    # Run experiments
    all_results = run_experiments(mssv, func_names, dims, pop_sizes)
    
    # Create convergence plots
    create_convergence_plots(all_results)
    
    # Create animations for 2D problems
    create_animations(all_results, mssv)
    
    # Create results tables
    tables = create_results_tables(all_results)
    
    return all_results, tables 