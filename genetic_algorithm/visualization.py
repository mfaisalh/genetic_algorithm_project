import matplotlib.pyplot as plt
from deap.tools import Logbook

def plot_fitness(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    line2 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    plt.show()
