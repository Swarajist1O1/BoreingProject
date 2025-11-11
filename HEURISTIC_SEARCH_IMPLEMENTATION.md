# Heuristic Search Algorithms Implementation

## Overview

This document describes the heuristic search algorithms implemented in the Groundwater Monitoring System. These algorithms are designed to optimize various aspects of groundwater resource management, including resource allocation, monitoring network design, and distribution path optimization.

## Implemented Algorithms

### 1. A* Search Algorithm (`AStarSearch`)

**Purpose**: Optimal pathfinding and resource distribution routing

**Key Features**:
- Manhattan and Euclidean distance heuristics
- 8-directional movement support
- Obstacle avoidance
- Optimal path guarantees

**Applications**:
- Water distribution network optimization
- Resource transportation routing
- Infrastructure planning paths

**Code Example**:
```python
from backend.heuristic_search import AStarSearch

astar = AStarSearch(grid_size=(50, 50))
path = astar.search(start=(0, 0), goal=(25, 25))
```

### 2. Genetic Algorithm (`GeneticAlgorithm`)

**Purpose**: Multi-variable optimization for resource allocation

**Key Features**:
- Tournament selection
- Single-point crossover
- Gaussian mutation
- Elitism preservation

**Parameters**:
- Population Size: 100
- Mutation Rate: 0.1
- Crossover Rate: 0.8
- Max Generations: 500

**Applications**:
- Well placement optimization
- Resource allocation across districts
- Multi-objective optimization scenarios

**Code Example**:
```python
from backend.heuristic_search import GeneticAlgorithm

ga = GeneticAlgorithm(population_size=100, max_generations=500)
solution, fitness = ga.search(
    num_variables=6,
    bounds=[(0, 100)] * 6,
    fitness_function=custom_fitness
)
```

### 3. Simulated Annealing (`SimulatedAnnealing`)

**Purpose**: Local optimization with escape from local minima

**Key Features**:
- Exponential cooling schedule
- Probabilistic acceptance
- Neighbor generation with bounds checking
- Temperature-based exploration

**Parameters**:
- Initial Temperature: 1000.0
- Cooling Rate: 0.95
- Minimum Temperature: 1e-8
- Max Iterations: 10,000

**Applications**:
- Management strategy refinement
- Parameter tuning
- Cost minimization problems

**Code Example**:
```python
from backend.heuristic_search import SimulatedAnnealing

sa = SimulatedAnnealing(initial_temp=1000.0, cooling_rate=0.95)
solution, cost = sa.search(
    initial_state=[10, 20, 30],
    bounds=[(0, 50), (0, 100), (0, 150)],
    cost_function=objective_function
)
```

### 4. Particle Swarm Optimization (`ParticleSwarmOptimization`)

**Purpose**: Swarm intelligence for global optimization

**Key Features**:
- Velocity-based particle movement
- Personal and global best tracking
- Inertia weight control
- Cognitive and social parameters

**Parameters**:
- Number of Particles: 30
- Max Iterations: 1000
- Inertia Weight (w): 0.729
- Cognitive Parameter (c1): 1.494
- Social Parameter (c2): 1.494

**Applications**:
- Monitoring network optimization
- Multi-dimensional parameter optimization
- Continuous optimization problems

**Code Example**:
```python
from backend.heuristic_search import ParticleSwarmOptimization

pso = ParticleSwarmOptimization(num_particles=30, max_iterations=1000)
solution, fitness = pso.search(
    num_dimensions=4,
    bounds=[(0, 100)] * 4,
    fitness_function=optimization_target
)
```

## Manager Class

### `HeuristicSearchManager`

The `HeuristicSearchManager` class provides a unified interface for all heuristic search algorithms and includes high-level optimization methods.

**Key Methods**:

#### `optimize_well_placement()`
Optimizes the placement of groundwater wells to maximize coverage of demand points.

**Parameters**:
- `num_wells`: Number of wells to place
- `region_bounds`: Geographical boundaries (x_min, y_min, x_max, y_max)
- `water_demand_points`: List of demand point coordinates
- `algorithm`: Algorithm to use ('genetic', 'pso', 'simulated_annealing')

**Returns**:
```python
{
    'algorithm': 'genetic',
    'wells': [(x1, y1), (x2, y2), ...],
    'fitness': 15.67,
    'coverage_score': 0.89,
    'optimization_metadata': {...}
}
```

#### `optimize_monitoring_network()`
Optimizes monitoring station placement for maximum risk area coverage.

**Parameters**:
- `grid_size`: Grid dimensions (width, height)
- `num_stations`: Number of monitoring stations
- `risk_areas`: List of high-risk area coordinates
- `algorithm`: Algorithm to use ('genetic', 'pso')

**Returns**:
```python
{
    'algorithm': 'pso',
    'stations': [(x1, y1), (x2, y2), ...],
    'fitness': 12.34,
    'coverage_efficiency': 0.85,
    'optimization_metadata': {...}
}
```

#### `find_optimal_path()`
Finds optimal path between two points using A* algorithm.

**Parameters**:
- `start`: Starting coordinates (x, y)
- `goal`: Goal coordinates (x, y)
- `obstacles`: List of obstacle coordinates
- `grid_size`: Grid dimensions

**Returns**:
```python
{
    'algorithm': 'astar',
    'path': [(x1, y1), (x2, y2), ...],
    'path_length': 15,
    'path_cost': 14,
    'success': True,
    'optimization_metadata': {...}
}
```

## Integration with Main System

### Chatbot Integration

The heuristic search algorithms are integrated into the chatbot system (`frontend/chatbot.py`) through the following mechanisms:

1. **Import and Initialization**:
```python
from heuristic_search import HeuristicSearchManager, WaterResourceState
self.heuristic_manager = HeuristicSearchManager()
```

2. **Data Preparation**:
```python
def _prepare_heuristic_data(self):
    # Convert groundwater data to heuristic search format
    # Initialize algorithms with district data
```

3. **Resource Optimization**:
```python
def optimize_resource_allocation(self, state: str) -> str:
    # Use heuristic algorithms for resource allocation
    # Return optimization results and recommendations
```

### App Integration

The main application (`frontend/app.py`) includes heuristic search capabilities:

```python
# Import heuristic search capabilities for advanced optimization
try:
    from heuristic_search import HeuristicSearchManager
    HEURISTIC_OPTIMIZATION_ENABLED = True
except ImportError:
    HEURISTIC_OPTIMIZATION_ENABLED = False
```

## Usage Examples

### Complete Optimization Workflow

```python
from backend.heuristic_search import HeuristicSearchManager

# Initialize manager
manager = HeuristicSearchManager()

# Define problem parameters
water_demand_points = [(10, 15), (25, 30), (40, 20)]
region_bounds = (0, 0, 50, 50)

# Optimize well placement
well_result = manager.optimize_well_placement(
    num_wells=3,
    region_bounds=region_bounds,
    water_demand_points=water_demand_points,
    algorithm='genetic'
)

print(f"Optimal wells: {well_result['wells']}")
print(f"Coverage score: {well_result['coverage_score']:.3f}")

# Optimize monitoring network
risk_areas = [(5, 5), (15, 25), (35, 15)]
monitoring_result = manager.optimize_monitoring_network(
    grid_size=(50, 50),
    num_stations=4,
    risk_areas=risk_areas,
    algorithm='pso'
)

print(f"Optimal stations: {monitoring_result['stations']}")
print(f"Coverage efficiency: {monitoring_result['coverage_efficiency']:.3f}")

# Get optimization summary
summary = manager.get_optimization_summary()
print(f"Total optimizations: {summary['total_optimizations']}")
```

### Chatbot Interaction Examples

Users can interact with the heuristic search system through the chatbot:

**Optimization Queries**:
- "Optimize resource allocation for Tamil Nadu"
- "Find optimal well placement strategy"
- "Heuristic optimization for water distribution"
- "Resource allocation optimization analysis"

**Expected Responses**:
The chatbot will use the heuristic search algorithms to provide optimization recommendations, including:
- Optimal resource allocation percentages
- Strategic recommendations
- Algorithm performance metrics
- Visualization suggestions

## Performance Considerations

### Algorithm Selection Guidelines

1. **A* Search**: Best for pathfinding and routing problems with clear start/goal states
2. **Genetic Algorithm**: Suitable for complex multi-variable optimization with discrete/continuous variables
3. **Simulated Annealing**: Good for local refinement and escaping local optima
4. **Particle Swarm Optimization**: Effective for continuous optimization problems with multiple objectives

### Computational Complexity

- **A* Search**: O(b^d) where b is branching factor, d is depth
- **Genetic Algorithm**: O(g × p × f) where g is generations, p is population size, f is fitness evaluation cost
- **Simulated Annealing**: O(i × f) where i is iterations, f is fitness evaluation cost
- **PSO**: O(i × p × f) where i is iterations, p is particles, f is fitness evaluation cost

### Memory Usage

- **A* Search**: O(b^d) for open/closed sets
- **Genetic Algorithm**: O(p × n) where p is population size, n is chromosome length
- **Simulated Annealing**: O(n) where n is solution size
- **PSO**: O(p × n) where p is particles, n is dimensions

## Future Enhancements

### Planned Algorithm Additions

1. **Ant Colony Optimization**: For path optimization problems
2. **Differential Evolution**: For parameter optimization
3. **Tabu Search**: For combinatorial optimization
4. **Multi-Objective Genetic Algorithm (NSGA-II)**: For multi-criteria optimization

### Integration Improvements

1. **Real-time Optimization**: Live optimization based on sensor data
2. **Distributed Computing**: Parallel algorithm execution
3. **Machine Learning Integration**: Adaptive parameter tuning
4. **Visualization Tools**: Interactive optimization result displays

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies (numpy, random, math) are installed
2. **Convergence Problems**: Adjust algorithm parameters (population size, iterations)
3. **Memory Issues**: Reduce problem size or use iterative approaches
4. **Performance Issues**: Consider algorithm selection based on problem characteristics

### Error Handling

The system includes comprehensive error handling:
- Boundary constraint enforcement
- Invalid parameter detection
- Graceful fallback when algorithms fail
- Detailed error reporting and logging

## Conclusion

The heuristic search implementation provides a comprehensive suite of optimization algorithms specifically tailored for groundwater resource management. The modular design allows for easy extension and customization while maintaining integration with the main application systems.

The algorithms demonstrate strong performance across various optimization scenarios and provide practical solutions for real-world groundwater management challenges.
