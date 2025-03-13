import numpy as np

class ObjectiveFunction:
    def __init__(self, name, dim, search_domain, global_minimum_value=0.0, global_minimum_position=None):
        self.name = name
        self.dim = dim
        self.search_domain = search_domain  # [lower_bound, upper_bound]
        self.global_minimum_value = global_minimum_value
        self.global_minimum_position = global_minimum_position if global_minimum_position is not None else np.zeros(dim)
        self.evaluations = 0
    
    def __call__(self, x):
        self.evaluations += 1
        return self._evaluate(x)
    
    def _evaluate(self, x):
        raise NotImplementedError("Subclasses must implement _evaluate method")
    
    def reset_evaluations(self):
        self.evaluations = 0


class Sphere(ObjectiveFunction):
    def __init__(self, dim):
        super().__init__(
            name="Sphere",
            dim=dim,
            search_domain=[-5.12, 5.12],
            global_minimum_value=0.0,
            global_minimum_position=np.zeros(dim)
        )
    
    def _evaluate(self, x):
        return np.sum(x**2)


class Griewank(ObjectiveFunction):
    def __init__(self, dim):
        super().__init__(
            name="Griewank",
            dim=dim,
            search_domain=[-600, 600],
            global_minimum_value=0.0,
            global_minimum_position=np.zeros(dim)
        )
    
    def _evaluate(self, x):
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return 1 + sum_part - prod_part


class Rosenbrock(ObjectiveFunction):
    def __init__(self, dim):
        super().__init__(
            name="Rosenbrock",
            dim=dim,
            search_domain=[-5, 10],
            global_minimum_value=0.0,
            global_minimum_position=np.ones(dim)
        )
    
    def _evaluate(self, x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


class Rastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super().__init__(
            name="Rastrigin",
            dim=dim,
            search_domain=[-5.12, 5.12],
            global_minimum_value=0.0,
            global_minimum_position=np.zeros(dim)
        )
    
    def _evaluate(self, x):
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Ackley(ObjectiveFunction):
    def __init__(self, dim):
        super().__init__(
            name="Ackley",
            dim=dim,
            search_domain=[-32.768, 32.768],
            global_minimum_value=0.0,
            global_minimum_position=np.zeros(dim)
        )
    
    def _evaluate(self, x):
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dim)
        return term1 + term2 + 20 + np.e


def get_function(func_name, dim):
    """
    Factory function to create objective function instances
    """
    func_map = {
        "sphere": Sphere,
        "griewank": Griewank,
        "rosenbrock": Rosenbrock,
        "rastrigin": Rastrigin,
        "ackley": Ackley
    }
    
    if func_name.lower() not in func_map:
        raise ValueError(f"Unknown function: {func_name}")
    
    return func_map[func_name.lower()](dim) 