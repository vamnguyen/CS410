import numpy as np
from tqdm import tqdm

class DifferentialEvolution:
    def __init__(self, objective_function, pop_size, F=0.8, CR=0.9, max_evals=None):
        """
        Initialize Differential Evolution algorithm
        
        Parameters:
        -----------
        objective_function : ObjectiveFunction
            The objective function to minimize
        pop_size : int
            Population size
        F : float
            Differential weight (mutation factor)
        CR : float
            Crossover probability
        max_evals : int
            Maximum number of function evaluations
        """
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.dim = objective_function.dim
        self.bounds = [objective_function.search_domain] * self.dim
        self.max_evals = max_evals if max_evals is not None else 2000 if self.dim == 2 else 10000
        
        # For tracking
        self.best_solutions = []
        self.best_fitnesses = []
        self.num_evaluations = []
        self.population_history = []
    
    def initialize_population(self, seed=None):
        """Initialize random population within bounds"""
        if seed is not None:
            np.random.seed(seed)
        
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        
        # Initialize population randomly within bounds
        self.population = lower_bounds + np.random.rand(self.pop_size, self.dim) * (upper_bounds - lower_bounds)
        
        # Evaluate initial population
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])
        
        # Track best solution
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        
        # Save initial state
        self.best_solutions.append(self.best_solution.copy())
        self.best_fitnesses.append(self.best_fitness)
        self.num_evaluations.append(self.objective_function.evaluations)
        if self.dim == 2:  # Only save population history for 2D problems (for visualization)
            self.population_history.append(self.population.copy())
    
    def optimize(self, seed=None, verbose=True):
        """Run the DE optimization process"""
        self.initialize_population(seed)
        
        generation = 0
        pbar = tqdm(total=self.max_evals, disable=not verbose)
        pbar.update(self.objective_function.evaluations)
        
        while self.objective_function.evaluations < self.max_evals:
            generation += 1
            
            for i in range(self.pop_size):
                # Select three random individuals, different from the current one
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Create mutant vector
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                # Apply bounds
                mutant = np.clip(mutant, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                
                # Crossover
                trial = np.zeros(self.dim)
                j_rand = np.random.randint(0, self.dim)
                
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = self.population[i][j]
                
                # Evaluate trial vector
                trial_fitness = self.objective_function(trial)
                
                # Selection
                if trial_fitness <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    # Update best solution if needed
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial.copy()
                        self.best_fitness = trial_fitness
                        self.best_idx = i
            
            # Save current state
            self.best_solutions.append(self.best_solution.copy())
            self.best_fitnesses.append(self.best_fitness)
            self.num_evaluations.append(self.objective_function.evaluations)
            
            if self.dim == 2:  # Only save population history for 2D problems
                self.population_history.append(self.population.copy())
            
            # Update progress bar
            pbar.update(self.objective_function.evaluations - pbar.n)
            
            # Early stopping if we've reached max evaluations
            if self.objective_function.evaluations >= self.max_evals:
                break
        
        pbar.close()
        return self.best_solution, self.best_fitness 