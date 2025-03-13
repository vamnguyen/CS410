import numpy as np
from tqdm import tqdm

class CrossEntropyMethod:
    def __init__(self, objective_function, pop_size, elite_ratio=0.2, alpha=0.7, max_evals=None):
        """
        Initialize Cross-Entropy Method algorithm
        
        Parameters:
        -----------
        objective_function : ObjectiveFunction
            The objective function to minimize
        pop_size : int
            Population size
        elite_ratio : float
            Ratio of elite samples to use for updating distribution
        alpha : float
            Smoothing parameter for distribution updates
        max_evals : int
            Maximum number of function evaluations
        """
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.elite_ratio = elite_ratio
        self.alpha = alpha
        self.dim = objective_function.dim
        self.bounds = [objective_function.search_domain] * self.dim
        self.max_evals = max_evals if max_evals is not None else 2000 if self.dim == 2 else 10000
        
        # Number of elite samples
        self.num_elite = max(1, int(self.pop_size * self.elite_ratio))
        
        # Initialize distribution parameters
        self.mean = np.zeros(self.dim)
        self.std = np.zeros(self.dim)
        
        # For tracking
        self.best_solutions = []
        self.best_fitnesses = []
        self.num_evaluations = []
        self.population_history = []
    
    def initialize_distribution(self, seed=None):
        """Initialize distribution parameters based on bounds"""
        if seed is not None:
            np.random.seed(seed)
        
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        
        # Initialize mean at center of bounds
        self.mean = (lower_bounds + upper_bounds) / 2
        
        # Initialize std to cover the entire search space
        self.std = (upper_bounds - lower_bounds) / 2
    
    def sample_population(self):
        """Sample population from current distribution"""
        population = np.random.normal(
            loc=self.mean.reshape(1, -1),
            scale=self.std.reshape(1, -1),
            size=(self.pop_size, self.dim)
        )
        
        # Apply bounds
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        population = np.clip(population, lower_bounds, upper_bounds)
        
        return population
    
    def update_distribution(self, population, fitness):
        """Update distribution parameters based on elite samples"""
        # Get indices of elite samples
        elite_indices = np.argsort(fitness)[:self.num_elite]
        elite_samples = population[elite_indices]
        
        # Calculate new distribution parameters
        new_mean = np.mean(elite_samples, axis=0)
        new_std = np.std(elite_samples, axis=0)
        
        # Apply smoothing
        self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        self.std = self.alpha * self.std + (1 - self.alpha) * new_std
    
    def optimize(self, seed=None, verbose=True):
        """Run the CEM optimization process"""
        self.initialize_distribution(seed)
        
        # Initialize best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        
        generation = 0
        pbar = tqdm(total=self.max_evals, disable=not verbose)
        
        while self.objective_function.evaluations < self.max_evals:
            generation += 1
            
            # Sample population
            population = self.sample_population()
            
            # Evaluate population
            fitness = np.array([self.objective_function(ind) for ind in population])
            
            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_solution = population[min_idx].copy()
                self.best_fitness = fitness[min_idx]
            
            # Update distribution
            self.update_distribution(population, fitness)
            
            # Save current state
            self.best_solutions.append(self.best_solution.copy())
            self.best_fitnesses.append(self.best_fitness)
            self.num_evaluations.append(self.objective_function.evaluations)
            
            if self.dim == 2:  # Only save population history for 2D problems
                self.population_history.append(population.copy())
            
            # Update progress bar
            pbar.update(self.objective_function.evaluations - pbar.n)
            
            # Early stopping if we've reached max evaluations
            if self.objective_function.evaluations >= self.max_evals:
                break
        
        pbar.close()
        return self.best_solution, self.best_fitness 