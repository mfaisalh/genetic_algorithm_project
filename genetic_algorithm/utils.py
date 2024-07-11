import json
import pickle
import logging

def save_results(best_individual, best_fitness, logbook, filename="results.json"):
    results = {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "logbook": logbook
    }
    with open(filename, 'w') as f:
        json.dump(results, f)
    logging.info(f"Results saved to {filename}")

def load_state(filename):
    with open(filename, 'rb') as f:
        pop, logbook, hof = pickle.load(f)
    return pop, logbook, hof

def save_state(pop, logbook, hof, filename):
    with open(filename, 'wb') as f:
        pickle.dump((pop, logbook, hof), f)
    logging.info(f"State saved to {filename}")
