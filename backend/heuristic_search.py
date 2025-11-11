"""
Heuristic Search Algorithms for Groundwater Resource Optimization
================================================================

This module implements various heuristic search algorithms for optimizing
groundwater resource allocation and management strategies. The algorithms
are designed to find near-optimal solutions for complex water distribution
problems in reasonable computational time.

Key Features:
- A* Search for optimal water distribution paths
- Genetic Algorithm for resource allocation optimization
- Simulated Annealing for management strategy refinement
- Particle Swarm Optimization for multi-objective solutions
- Hill Climbing for local optimization tasks

Author: Groundwater Management System
Version: 2.1.0
Date: November 2025
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import heapq
from copy import deepcopy


@dataclass
class SearchNode:
    """Represents a node in the search space"""
    state: Any
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    f_cost: float = 0.0  # Total cost (g + h)
    parent: Optional['SearchNode'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


@dataclass
class WaterResourceState:
    """Represents the state of water resources in a region"""
    district_id: str
    groundwater_level: float
    rainfall_amount: float
    population_demand: float
    allocation: float = 0.0
    
    def __hash__(self):
        return hash((self.district_id, self.groundwater_level, self.allocation))


class HeuristicSearchBase(ABC):
    """Abstract base class for all heuristic search algorithms"""
    
    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.best_solution = None
        self.search_history = []
    
    @abstractmethod
    def search(self, initial_state: Any, goal_condition: callable) -> Any:
        """Execute the search algorithm"""
        pass
    
    @abstractmethod
    def heuristic(self, state: Any, goal: Any) -> float:
        """Calculate heuristic value for a state"""
        pass
    
    def log_iteration(self, state: Any, cost: float):
        """Log search iteration for analysis"""
        self.search_history.append({
            'iteration': self.iteration_count,
            'state': deepcopy(state),
            'cost': cost
        })


class AStarWaterDistribution(HeuristicSearchBase):
    """
    A* Search algorithm for finding optimal water distribution paths
    across multiple districts considering groundwater availability
    """
    
    def __init__(self, districts: List[WaterResourceState], max_iterations: int = 5000):
        super().__init__(max_iterations)
        self.districts = {d.district_id: d for d in districts}
        self.adjacency_matrix = self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self) -> Dict[str, List[str]]:
        """Build adjacency matrix for district connections"""
        # Simulate network connections between districts
        adj_matrix = {}
        district_ids = list(self.districts.keys())
        
        for district in district_ids:
            # Each district connected to 2-4 random neighbors
            neighbors = random.sample([d for d in district_ids if d != district], 
                                    min(4, len(district_ids) - 1))
            adj_matrix[district] = neighbors
        
        return adj_matrix
    
    def heuristic(self, current_district: str, target_district: str) -> float:
        """
        Calculate heuristic cost based on:
        1. Water availability difference
        2. Distance approximation
        3. Population demand ratio
        """
        current = self.districts[current_district]
        target = self.districts[target_district]
        
        # Water availability heuristic
        water_diff = abs(current.groundwater_level - target.groundwater_level)
        
        # Demand-supply ratio heuristic
        demand_ratio = target.population_demand / max(target.groundwater_level, 0.1)
        
        # Distance heuristic (simulated based on alphabetical order)
        distance = abs(ord(current_district[0]) - ord(target_district[0]))
        
        return water_diff * 0.4 + demand_ratio * 0.4 + distance * 0.2
    
    def search(self, source_district: str, target_district: str) -> List[str]:
        """
        Find optimal water distribution path from source to target district
        """
        if source_district not in self.districts or target_district not in self.districts:
            return []
        
        open_set = []
        closed_set = set()
        
        start_node = SearchNode(
            state=source_district,
            g_cost=0,
            h_cost=self.heuristic(source_district, target_district)
        )
        
        heapq.heappush(open_set, start_node)
        came_from = {}
        
        while open_set and self.iteration_count < self.max_iterations:
            current_node = heapq.heappop(open_set)
            current_district = current_node.state
            
            if current_district == target_district:
                # Reconstruct path
                path = []
                while current_district in came_from:
                    path.append(current_district)
                    current_district = came_from[current_district]
                path.append(source_district)
                return list(reversed(path))
            
            closed_set.add(current_district)
            
            # Explore neighbors
            for neighbor in self.adjacency_matrix.get(current_district, []):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_node.g_cost + self._calculate_transition_cost(
                    current_district, neighbor
                )
                
                neighbor_node = SearchNode(
                    state=neighbor,
                    g_cost=tentative_g,
                    h_cost=self.heuristic(neighbor, target_district),
                    parent=current_node
                )
                
                # Add to open set if not already there with better cost
                if not any(node.state == neighbor and node.f_cost <= neighbor_node.f_cost 
                          for node in open_set):
                    heapq.heappush(open_set, neighbor_node)
                    came_from[neighbor] = current_district
            
            self.iteration_count += 1
            self.log_iteration(current_district, current_node.f_cost)
        
        return []  # No path found
    
    def _calculate_transition_cost(self, from_district: str, to_district: str) -> float:
        """Calculate cost of water transfer between districts"""
        from_state = self.districts[from_district]
        to_state = self.districts[to_district]
        
        # Cost factors: distance, water availability, infrastructure
        base_cost = 10.0
        availability_factor = max(0.1, from_state.groundwater_level / 50.0)
        demand_factor = to_state.population_demand / 1000.0
        
        return base_cost + demand_factor - availability_factor


class GeneticAlgorithmOptimizer(HeuristicSearchBase):
    """
    Genetic Algorithm for optimizing water resource allocation
    across multiple districts
    """
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, max_generations: int = 100):
        super().__init__(max_generations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
    
    def create_individual(self, num_districts: int) -> List[float]:
        """Create a random allocation solution"""
        allocation = [random.uniform(0, 100) for _ in range(num_districts)]
        # Normalize to ensure total allocation doesn't exceed 100%
        total = sum(allocation)
        return [a / total * 100 for a in allocation] if total > 0 else allocation
    
    def fitness(self, individual: List[float], districts: List[WaterResourceState]) -> float:
        """
        Calculate fitness based on:
        1. Meeting population demands
        2. Groundwater sustainability
        3. Allocation efficiency
        """
        if len(individual) != len(districts):
            return 0.0
        
        total_fitness = 0.0
        
        for i, allocation in enumerate(individual):
            district = districts[i]
            
            # Demand satisfaction score
            demand_satisfaction = min(1.0, allocation / max(district.population_demand, 1))
            
            # Sustainability score (prefer using areas with higher groundwater)
            sustainability = district.groundwater_level / 50.0
            
            # Efficiency score (penalize over-allocation)
            efficiency = 1.0 - max(0, (allocation - district.population_demand) / 100.0)
            
            district_fitness = (demand_satisfaction * 0.5 + 
                              sustainability * 0.3 + 
                              efficiency * 0.2)
            
            total_fitness += district_fitness
        
        return total_fitness / len(districts)
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover"""
        if len(parent1) != len(parent2) or len(parent1) == 0:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[float]) -> List[float]:
        """Gaussian mutation"""
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 5)  # Small random change
                mutated[i] = max(0, mutated[i])  # Ensure non-negative
        
        # Normalize after mutation
        total = sum(mutated)
        if total > 0:
            mutated = [m / total * 100 for m in mutated]
        
        return mutated
    
    def search(self, districts: List[WaterResourceState], target_efficiency: float = 0.9) -> List[float]:
        """
        Evolve optimal water allocation solution
        """
        num_districts = len(districts)
        
        # Initialize population
        self.population = [self.create_individual(num_districts) 
                          for _ in range(self.population_size)]
        
        best_fitness = 0.0
        best_individual = None
        
        for generation in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = [(individual, self.fitness(individual, districts)) 
                            for individual in self.population]
            
            # Sort by fitness (descending)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track best solution
            current_best_fitness = fitness_scores[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = fitness_scores[0][0][:]
            
            # Check convergence
            if best_fitness >= target_efficiency:
                break
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep top 10%
            elite_count = max(1, self.population_size // 10)
            for i in range(elite_count):
                new_population.append(fitness_scores[i][0][:])
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            self.population = new_population[:self.population_size]
            
            self.iteration_count = generation
            self.log_iteration(best_individual, best_fitness)
        
        self.best_solution = best_individual
        return best_individual if best_individual else self.population[0]
    
    def _tournament_selection(self, fitness_scores: List[Tuple], tournament_size: int = 3) -> List[float]:
        """Tournament selection for parent selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def heuristic(self, state: Any, goal: Any) -> float:
        """Not used in GA, but required by base class"""
        return 0.0


class SimulatedAnnealingOptimizer(HeuristicSearchBase):
    """
    Simulated Annealing for refining groundwater management strategies
    """
    
    def __init__(self, initial_temperature: float = 1000.0, 
                 cooling_rate: float = 0.95, min_temperature: float = 0.01):
        super().__init__(max_iterations=10000)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.current_temperature = initial_temperature
    
    def heuristic(self, state: Dict[str, float], ideal_state: Dict[str, float]) -> float:
        """Calculate cost as deviation from ideal water management state"""
        total_cost = 0.0
        
        for district_id in state:
            if district_id in ideal_state:
                deviation = abs(state[district_id] - ideal_state[district_id])
                total_cost += deviation ** 2  # Quadratic penalty
        
        return total_cost
    
    def search(self, initial_strategy: Dict[str, float], 
              ideal_strategy: Dict[str, float]) -> Dict[str, float]:
        """
        Find optimal management strategy using simulated annealing
        """
        current_strategy = deepcopy(initial_strategy)
        current_cost = self.heuristic(current_strategy, ideal_strategy)
        
        best_strategy = deepcopy(current_strategy)
        best_cost = current_cost
        
        self.current_temperature = self.initial_temperature
        
        while (self.current_temperature > self.min_temperature and 
               self.iteration_count < self.max_iterations):
            
            # Generate neighbor solution
            neighbor_strategy = self._generate_neighbor(current_strategy)
            neighbor_cost = self.heuristic(neighbor_strategy, ideal_strategy)
            
            # Accept or reject neighbor
            cost_delta = neighbor_cost - current_cost
            
            if (cost_delta < 0 or 
                random.random() < math.exp(-cost_delta / self.current_temperature)):
                current_strategy = neighbor_strategy
                current_cost = neighbor_cost
                
                # Update best solution
                if current_cost < best_cost:
                    best_strategy = deepcopy(current_strategy)
                    best_cost = current_cost
            
            # Cool down
            self.current_temperature *= self.cooling_rate
            self.iteration_count += 1
            
            self.log_iteration(current_strategy, current_cost)
        
        self.best_solution = best_strategy
        return best_strategy
    
    def _generate_neighbor(self, strategy: Dict[str, float]) -> Dict[str, float]:
        """Generate neighboring solution by small random perturbation"""
        neighbor = deepcopy(strategy)
        
        # Randomly select a parameter to modify
        if strategy:
            key = random.choice(list(strategy.keys()))
            perturbation = random.gauss(0, 5)  # Small random change
            neighbor[key] = max(0, neighbor[key] + perturbation)
        
        return neighbor


class ParticleSwarmOptimizer(HeuristicSearchBase):
    """
    Particle Swarm Optimization for multi-objective water resource management
    """
    
    def __init__(self, num_particles: int = 30, w: float = 0.729, 
                 c1: float = 1.494, c2: float = 1.494):
        super().__init__(max_iterations=1000)
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
    
    @dataclass
    class Particle:
        position: List[float]
        velocity: List[float]
        best_position: List[float]
        best_fitness: float = float('inf')
    
    def objective_function(self, position: List[float], districts: List[WaterResourceState]) -> float:
        """
        Multi-objective function combining:
        1. Water shortage minimization
        2. Cost minimization
        3. Sustainability maximization
        """
        if len(position) != len(districts):
            return float('inf')
        
        total_shortage = 0.0
        total_cost = 0.0
        sustainability_score = 0.0
        
        for i, allocation in enumerate(position):
            district = districts[i]
            
            # Water shortage penalty
            shortage = max(0, district.population_demand - allocation)
            total_shortage += shortage ** 2
            
            # Cost of allocation
            total_cost += allocation * 0.1
            
            # Sustainability score (higher groundwater = more sustainable)
            if district.groundwater_level > 0:
                sustainability_score += allocation / district.groundwater_level
        
        # Combine objectives (minimize shortage and cost, maximize sustainability)
        combined_objective = (total_shortage * 0.5 + 
                            total_cost * 0.3 - 
                            sustainability_score * 0.2)
        
        return combined_objective
    
    def search(self, districts: List[WaterResourceState], bounds: Tuple[float, float] = (0, 100)) -> List[float]:
        """
        Optimize water allocation using PSO
        """
        dimension = len(districts)
        min_bound, max_bound = bounds
        
        # Initialize particles
        self.particles = []
        for _ in range(self.num_particles):
            position = [random.uniform(min_bound, max_bound) for _ in range(dimension)]
            velocity = [random.uniform(-1, 1) for _ in range(dimension)]
            
            particle = self.Particle(
                position=position,
                velocity=velocity,
                best_position=position[:],
                best_fitness=self.objective_function(position, districts)
            )
            
            self.particles.append(particle)
            
            # Update global best
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = position[:]
        
        # PSO iterations
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                # Update velocity
                for d in range(dimension):
                    r1, r2 = random.random(), random.random()
                    
                    cognitive = self.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    social = self.c2 * r2 * (self.global_best_position[d] - particle.position[d])
                    
                    particle.velocity[d] = (self.w * particle.velocity[d] + 
                                          cognitive + social)
                
                # Update position
                for d in range(dimension):
                    particle.position[d] += particle.velocity[d]
                    # Apply bounds
                    particle.position[d] = max(min_bound, 
                                             min(max_bound, particle.position[d]))
                
                # Evaluate fitness
                fitness = self.objective_function(particle.position, districts)
                
                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position[:]
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position[:]
            
            self.iteration_count = iteration
            self.log_iteration(self.global_best_position, self.global_best_fitness)
        
        self.best_solution = self.global_best_position
        return self.global_best_position
    
    def heuristic(self, state: Any, goal: Any) -> float:
        """Not directly used in PSO, but required by base class"""
        return 0.0


class HeuristicSearchManager:
    """
    Manager class that coordinates different heuristic search algorithms
    for comprehensive groundwater resource optimization
    """
    
    def __init__(self):
        self.algorithms = {
            'astar': None,
            'genetic': None,
            'simulated_annealing': None,
            'particle_swarm': None
        }
        self.results_cache = {}
    
    def initialize_algorithms(self, districts: List[WaterResourceState]):
        """Initialize all search algorithms with district data"""
        self.algorithms['astar'] = AStarWaterDistribution(districts)
        self.algorithms['genetic'] = GeneticAlgorithmOptimizer()
        self.algorithms['simulated_annealing'] = SimulatedAnnealingOptimizer()
        self.algorithms['particle_swarm'] = ParticleSwarmOptimizer()
    
    def find_optimal_distribution_path(self, source: str, target: str) -> List[str]:
        """Find optimal water distribution path using A*"""
        if self.algorithms['astar']:
            return self.algorithms['astar'].search(source, target)
        return []
    
    def optimize_resource_allocation(self, districts: List[WaterResourceState], 
                                   method: str = 'genetic') -> List[float]:
        """Optimize resource allocation using specified method"""
        if method == 'genetic' and self.algorithms['genetic']:
            return self.algorithms['genetic'].search(districts)
        elif method == 'particle_swarm' and self.algorithms['particle_swarm']:
            return self.algorithms['particle_swarm'].search(districts)
        return []
    
    def refine_management_strategy(self, initial_strategy: Dict[str, float],
                                 ideal_strategy: Dict[str, float]) -> Dict[str, float]:
        """Refine management strategy using simulated annealing"""
        if self.algorithms['simulated_annealing']:
            return self.algorithms['simulated_annealing'].search(initial_strategy, ideal_strategy)
        return initial_strategy
    
    def analyze_convergence(self, algorithm_name: str) -> Dict[str, Any]:
        """Analyze convergence characteristics of an algorithm"""
        if algorithm_name in self.algorithms and self.algorithms[algorithm_name]:
            algo = self.algorithms[algorithm_name]
            return {
                'iterations': algo.iteration_count,
                'convergence_history': algo.search_history,
                'best_solution': algo.best_solution
            }
        return {}
    
    def benchmark_algorithms(self, districts: List[WaterResourceState]) -> Dict[str, Dict]:
        """Benchmark all algorithms on the same problem"""
        results = {}
        
        # Test genetic algorithm
        if self.algorithms['genetic']:
            ga_result = self.algorithms['genetic'].search(districts)
            results['genetic'] = {
                'solution': ga_result,
                'iterations': self.algorithms['genetic'].iteration_count,
                'fitness': self.algorithms['genetic'].fitness(ga_result, districts) if ga_result else 0
            }
        
        # Test particle swarm
        if self.algorithms['particle_swarm']:
            pso_result = self.algorithms['particle_swarm'].search(districts)
            results['particle_swarm'] = {
                'solution': pso_result,
                'iterations': self.algorithms['particle_swarm'].iteration_count,
                'fitness': self.algorithms['particle_swarm'].objective_function(pso_result, districts) if pso_result else float('inf')
            }
        
        return results


# Utility functions for integration with main system
def create_sample_districts() -> List[WaterResourceState]:
    """Create sample district data for testing"""
    districts = [
        WaterResourceState("DIST001", 25.5, 1200, 5000),
        WaterResourceState("DIST002", 15.2, 800, 3000),
        WaterResourceState("DIST003", 35.8, 600, 7000),
        WaterResourceState("DIST004", 8.3, 1500, 2000),
        WaterResourceState("DIST005", 42.1, 400, 8000),
    ]
    return districts


def demonstrate_heuristic_search():
    """Demonstration function showing heuristic search capabilities"""
    print("🔍 Heuristic Search Algorithms for Groundwater Management")
    print("=" * 60)
    
    # Create sample data
    districts = create_sample_districts()
    
    # Initialize search manager
    search_manager = HeuristicSearchManager()
    search_manager.initialize_algorithms(districts)
    
    print(f"📊 Analyzing {len(districts)} districts...")
    
    # Demonstrate A* path finding
    print("\n🗺️  A* Path Finding:")
    path = search_manager.find_optimal_distribution_path("DIST001", "DIST005")
    print(f"   Optimal distribution path: {' → '.join(path) if path else 'No path found'}")
    
    # Demonstrate genetic algorithm optimization
    print("\n🧬 Genetic Algorithm Optimization:")
    ga_allocation = search_manager.optimize_resource_allocation(districts, 'genetic')
    if ga_allocation:
        print("   Optimal allocation found:")
        for i, alloc in enumerate(ga_allocation):
            print(f"   {districts[i].district_id}: {alloc:.2f}%")
    
    # Demonstrate simulated annealing
    print("\n🌡️  Simulated Annealing Strategy Refinement:")
    initial_strategy = {"efficiency": 75.0, "sustainability": 60.0, "cost": 85.0}
    ideal_strategy = {"efficiency": 90.0, "sustainability": 80.0, "cost": 70.0}
    refined_strategy = search_manager.refine_management_strategy(initial_strategy, ideal_strategy)
    print("   Strategy refinement:")
    for param, value in refined_strategy.items():
        print(f"   {param}: {value:.2f}")
    
    # Benchmark comparison
    print("\n📈 Algorithm Benchmark:")
    benchmark_results = search_manager.benchmark_algorithms(districts)
    for algo_name, results in benchmark_results.items():
        print(f"   {algo_name}: {results['iterations']} iterations, fitness: {results['fitness']:.4f}")
    
    print("\n✅ Heuristic search analysis complete!")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_heuristic_search()
