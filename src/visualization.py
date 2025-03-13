import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import os
import seaborn as sns
from scipy import stats

def plot_convergence(de_results, cem_results, func_name, dim, pop_size, save_path=None):
    """
    Plot convergence graph comparing DE and CEM
    
    Parameters:
    -----------
    de_results : list of dicts
        Results from DE runs
    cem_results : list of dicts
        Results from CEM runs
    func_name : str
        Name of the objective function
    dim : int
        Dimension of the problem
    pop_size : int
        Population size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    de_evals = de_results[0]['num_evaluations']
    de_fitnesses = np.array([r['best_fitnesses'] for r in de_results])
    cem_evals = cem_results[0]['num_evaluations']
    cem_fitnesses = np.array([r['best_fitnesses'] for r in cem_results])
    
    # Calculate mean and std
    de_mean = np.mean(de_fitnesses, axis=0)
    de_std = np.std(de_fitnesses, axis=0)
    cem_mean = np.mean(cem_fitnesses, axis=0)
    cem_std = np.std(cem_fitnesses, axis=0)
    
    # Plot mean
    plt.plot(de_evals, de_mean, 'b-', label='DE')
    plt.plot(cem_evals, cem_mean, 'r-', label='CEM')
    
    # Plot error bars (fill between)
    plt.fill_between(de_evals, de_mean - de_std, de_mean + de_std, color='b', alpha=0.2)
    plt.fill_between(cem_evals, cem_mean - cem_std, cem_mean + cem_std, color='r', alpha=0.2)
    
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value')
    plt.title(f'{func_name} (d={dim}, N={pop_size})')
    plt.legend()
    plt.grid(True)
    
    # Use log scale for y-axis if values are positive
    if np.all(de_mean > 0) and np.all(cem_mean > 0):
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_contour_animation(algorithm_name, obj_func, history, best_solutions, seed, save_path):
    """
    Create animation of population movement on contour plot
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the algorithm (DE or CEM)
    obj_func : ObjectiveFunction
        The objective function
    history : list of arrays
        History of populations
    best_solutions : list of arrays
        History of best solutions
    seed : int
        Random seed used
    save_path : str
        Path to save the animation
    """
    # Create grid for contour plot
    x_min, x_max = obj_func.search_domain
    y_min, y_max = obj_func.search_domain
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate function values on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj_func._evaluate(np.array([X[i, j], Y[i, j]]))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contour
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax)
    
    # Plot global optimum
    global_opt = obj_func.global_minimum_position
    ax.plot(global_opt[0], global_opt[1], 'r*', markersize=15, label='Global Optimum')
    
    # Initialize scatter plot for population
    population_scatter = ax.scatter([], [], c='white', edgecolor='black', s=50, label='Population')
    best_scatter = ax.scatter([], [], c='red', s=100, label='Best Solution')
    
    # Set title and labels
    ax.set_title(f'{algorithm_name} on {obj_func.name} (d=2, seed={seed})')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    
    # Animation update function
    def update(frame):
        population = history[frame]
        best_solution = best_solutions[frame]
        
        population_scatter.set_offsets(population[:, :2])
        best_scatter.set_offsets([best_solution[:2]])
        
        ax.set_title(f'{algorithm_name} on {obj_func.name} (d=2, seed={seed}, frame={frame})')
        return population_scatter, best_scatter
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=100, blit=True
    )
    
    # Save animation
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()

def create_results_table(results, func_name, dim):
    """
    Create a table of results for a specific function and dimension
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for different population sizes
    func_name : str
        Name of the objective function
    dim : int
        Dimension of the problem
    
    Returns:
    --------
    table : pandas.DataFrame
        Table of results
    """
    import pandas as pd
    
    # Create empty table
    table = pd.DataFrame(
        index=[8, 16, 32, 64, 128],
        columns=['DE_mean', 'DE_std', 'CEM_mean', 'CEM_std', 'p_value', 'better']
    )
    
    # Fill table
    for pop_size in table.index:
        de_fitnesses = np.array([r['best_fitness'] for r in results[pop_size]['de']])
        cem_fitnesses = np.array([r['best_fitness'] for r in results[pop_size]['cem']])
        
        de_mean = np.mean(de_fitnesses)
        de_std = np.std(de_fitnesses)
        cem_mean = np.mean(cem_fitnesses)
        cem_std = np.std(cem_fitnesses)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(de_fitnesses, cem_fitnesses, equal_var=False)
        
        # Determine which algorithm is better
        if p_value < 0.05:
            better = 'DE' if de_mean < cem_mean else 'CEM'
        else:
            better = 'None'
        
        # Fill row
        table.loc[pop_size] = [de_mean, de_std, cem_mean, cem_std, p_value, better]
    
    return table

def format_table_for_latex(table, func_name, dim):
    """Format table for LaTeX with bold for better algorithm"""
    latex_table = table.copy()
    
    # Format means and stds
    for col in ['DE_mean', 'DE_std', 'CEM_mean', 'CEM_std']:
        latex_table[col] = latex_table[col].apply(lambda x: f"{x:.2e}")
    
    # Format p-values
    latex_table['p_value'] = latex_table['p_value'].apply(lambda x: f"{x:.4f}")
    
    # Bold the better algorithm
    for idx in latex_table.index:
        if latex_table.loc[idx, 'better'] == 'DE':
            latex_table.loc[idx, 'DE_mean'] = f"\\textbf{{{latex_table.loc[idx, 'DE_mean']}}}"
        elif latex_table.loc[idx, 'better'] == 'CEM':
            latex_table.loc[idx, 'CEM_mean'] = f"\\textbf{{{latex_table.loc[idx, 'CEM_mean']}}}"
    
    # Drop the 'better' column
    latex_table = latex_table.drop(columns=['better'])
    
    # Rename columns
    latex_table.columns = ['DE Mean', 'DE Std', 'CEM Mean', 'CEM Std', 'p-value']
    
    return latex_table 