import random
import numpy as np
import logging
from deap import base, creator, tools, algorithms
from genetic_algorithm.visualization import plot_fitness
from genetic_algorithm.utils import save_results, load_state, save_state
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Atur logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneticAlgorithmML:
    def __init__(self, population_size=100, chromosome_length=5, generations=50, cxpb=0.5, mutpb=0.2, tournsize=3, elitism=True, save_interval=10, eval_func='sphere', elite_size=1):
        if population_size <= 0 or chromosome_length <= 0 or generations <= 0:
            raise ValueError("Parameters population_size, chromosome_length, and generations must be greater than zero.")
        if not (0 <= cxpb <= 1) or not (0 <= mutpb <= 1):
            raise ValueError("Parameters cxpb and mutpb must be between 0 and 1.")
        if tournsize <= 0:
            raise ValueError("Parameter tournsize must be greater than zero.")
        if elite_size <= 0:
            raise ValueError("Parameter elite_size must be greater than zero.")
        
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.elitism = elitism
        self.save_interval = save_interval
        self.elite_size = elite_size
        
        # Pilih fungsi evaluasi
        self.eval_func = self.select_eval_func(eval_func)
        
        self.toolbox = self.setup_toolbox()
        self.pop = self.toolbox.population(n=self.population_size)
        self.hof = tools.HallOfFame(self.elite_size)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.logbook = tools.Logbook()

        # Model Machine Learning untuk evaluasi
        self.ml_model = RandomForestRegressor()

    def setup_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -5, 5)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=self.chromosome_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", self.eval_func)
        
        return toolbox

    def select_eval_func(self, eval_func):
        if eval_func == 'sphere':
            return self.sphere
        elif eval_func == 'rastrigin':
            return self.rastrigin
        elif eval_func == 'rosenbrock':
            return self.rosenbrock
        else:
            raise ValueError("Unsupported evaluation function. Supported functions: 'sphere', 'rastrigin', 'rosenbrock'")

    def sphere(self, individual):
        return sum(x**2 for x in individual),

    def rastrigin(self, individual):
        A = 10
        return A * len(individual) + sum((x**2 - A * np.cos(2 * np.pi * x)) for x in individual),

    def rosenbrock(self, individual):
        return sum(100 * (individual[i+1] - individual[i]**2)**2 + (1 - individual[i])**2 for i in range(len(individual)-1)),

    def adaptive_mutation_rate(self, gen, initial_rate=0.2, final_rate=0.01):
        return initial_rate * ((final_rate / initial_rate) ** (gen / self.generations))

    def adaptive_crossover_rate(self, gen, initial_rate=0.5, final_rate=0.9):
        return initial_rate + (final_rate - initial_rate) * (gen / self.generations)

    def hybrid_local_search(self, individual, iterations=5):
        for _ in range(iterations):
            neighbor = tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)[0]
            neighbor.fitness.values = self.toolbox.evaluate(neighbor)
            if neighbor.fitness.values < individual.fitness.values:
                individual[:] = neighbor
                individual.fitness.values = neighbor.fitness.values
        return individual

    def evolve(self):
        logging.info("Memulai proses evolusi")
        pool = multiprocessing.Pool()
        self.toolbox.register("map", pool.map)
        
        # Data untuk pelatihan model ML
        X_train, y_train = [], []
        
        for gen in range(self.generations):
            self.cxpb = self.adaptive_crossover_rate(gen)
            self.mutpb = self.adaptive_mutation_rate(gen)

            offspring = algorithms.varAnd(self.pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                X_train.append(ind)
                y_train.append(fit[0])

            # Melatih model ML setiap beberapa generasi
            if gen % 5 == 0 and gen > 0:
                self.ml_model.fit(X_train, y_train)
                preds = self.ml_model.predict([ind for ind in offspring])
                mse = mean_squared_error(y_train, preds)
                logging.info(f"Generasi {gen}: MSE model ML = {mse}")
                
                for ind, pred in zip(offspring, preds):
                    ind.fitness.values = (pred,)

            for ind in offspring:
                ind = self.hybrid_local_search(ind)

            if self.elitism:
                elite = tools.selBest(self.pop, self.elite_size)
                self.pop = self.toolbox.select(offspring, k=len(self.pop) - self.elite_size) + elite
            else:
                self.pop = self.toolbox.select(offspring, k=len(self.pop))
            
            record = self.stats.compile(self.pop)
            self.logbook.record(gen=gen, **record)
            self.hof.update(self.pop)
            
            logging.info(f"Generasi {gen}: Fitness terbaik = {record['min']}")

            if gen % self.save_interval == 0:
                save_state(self.pop, self.logbook, self.hof, filename=f"checkpoint_gen_{gen}.pkl")
        
        pool.close()
        pool.join()

        plot_fitness(self.logbook)
        logging.info("Proses evolusi selesai")
        return self.pop, self.logbook, self.hof

    def get_best_individual(self):
        return self.hof[0], self.hof[0].fitness.values

    def save_results(self, filename="results.json"):
        best_individual, best_fitness = self.get_best_individual()
        save_results(best_individual, best_fitness, self.logbook, filename)
        logging.info(f"Results saved to {filename}")

    def load_state(self, filename):
        self.pop, self.logbook, self.hof = load_state(filename)
        logging.info(f"State loaded from {filename}")

def main():
    # Definisikan fungsi evaluasi
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ga = GeneticAlgorithmML(eval_func='rastrigin', elite_size=3)
    #ga.load_state("checkpoint_gen_40.pkl")  # Memuat keadaan dari checkpoint
    pop, log, hof = ga.evolve()
    best_individual, best_fitness = ga.get_best_individual()
    
    print("Individu terbaik adalah:", best_individual)
    print("Dengan nilai fitness:", best_fitness)
    
    ga.save_results()

if __name__ == "__main__":
    main()
