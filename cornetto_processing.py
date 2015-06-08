#total cornetto is equivalent to whole ellipse poincare plot
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sd1s = []
    sd2s = []
    with open("total_cornettos.csv") as cornetto_file:
        reader = csv.reader(cornetto_file)
        for row in reader:
            sd1, sd2 = float(row[-2]), float(row[-1])
            sd1s.append(sd1)
            sd2s.append(sd2)
    print sd1s, sd2s
    plt.hist(sd1s)
    plt.xlabel("standard deviation")
    plt.ylabel("frequency")
    plt.savefig("sd1_histogram")
    plt.close()
    plt.hist(sd2s)
    plt.xlabel("standard deviation")
    plt.ylabel("frequency")
    plt.savefig("sd2_histogram")
    plt.close()
