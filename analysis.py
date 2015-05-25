import csv
import os.path
import glob
import math
import functools
import itertools
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mat_mlab
import numpy.fft as np_f
import numpy.linalg as np_l
import numpy as np
import pandas as pd
import pandas.tools.plotting as pd_plot
import scipy.signal as sci_sig
from hilbert_test import * #pollute that namespace woo
from plots import *

def coherence_over_time(west, north, name):
    west = np.array(west)
    north = np.array(north)
    to_plot = zip(west, north) #let's have transform
    cxy, f = plt.cohere(west, north) #for freqs first
    with open("./coherence_stats/" + name, "w") as stats_file:
        stats_file.write(",".join(map(str, f)))
        stats_file.write("\n")
        stats_file.write(",".join(map(str, cxy)))

def prob_dict(data):
    data = list(data)
    alphabet = set(data)
    probdict = {}
    for symbol in alphabet:
        ctr = sum(1 for x in data if x == symbol)
        probdict[symbol] = (float(ctr) / len(data))
    return probdict

def entropy(data):
    data_probs = prob_dict(data)
    return sum(-p * np.log2(p) for _, p in data_probs.iteritems())

def joint_entropy(data1, data2):
    probs = []
    data1, data2 = np.array(data1, dtype=np.int16), np.array(data2, dtype=np.int16)
    #should look at the sparsity of things
    for c1 in set(data1):
        for c2 in set(data2):
            probs.append(np.mean(np.logical_and(data1 == c1, data2 == c2)))
    probs = filter(lambda x: x != 0.0, probs)
    return np.sum(-p * np.log2(p) for p in probs)

def mutual_information(data1, data2):
    data1, data2 = list(data1), list(data2)
    return entropy(data1) + entropy(data2) - joint_entropy(data1, data2)

def auto_mutual_information(data, stepsize, stepmax):
    return cross_mutual_information(data, data, stepsize, stepmax)

def cross_mutual_information(data1, data2, stepsize, stepmax):
    first = np.array(data1)
    cmis = []
    for lag in xrange(0, stepmax, stepsize):
        lagged = np.roll(data2, -lag)
        cmis.append(mutual_information(first, lagged))
    return cmis

def discretize_data(data, bucket_size=0.2):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    buckets = np.linspace(data_min, data_max, num=64)
    idx = np.digitize(data, buckets)
    return buckets[idx-1], idx

def total_amis(data, name, num_points=10):
    with open("./ami_csvs/" + name + ".csv", "w") as total_ami:
        for idx, ami in enumerate(auto_mutual_information(data, num_points, len(data))):
            total_ami.write(str(num_points) + " " + str(ami))

def ami_plot(data, name, stepsize=1, max_bits=10, stepmax=50):
    digitized, idx = discretize_data(data)
    autos = auto_mutual_information(idx, stepsize, stepmax)
    xmin, xmax, ymin, ymax = 0, stepmax, 0, max_bits
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    ax.axis([xmin, xmax, ymin, ymax])
    plt.title(name)
    ax.plot(range(0, stepmax, stepsize), autos)
    #ranges
    plt.xlabel("lag")
    plt.ylabel("ami (bits)")
    plt.savefig("./ami_plots/" + name)

def total_cmis(west, north, name, num_points=10):
    with open("./cmi_csvs/" + name + ".csv", "w") as total_cmi:
        for idx, cmi in enumerate(cross_mutual_information(west, north, num_points, len(west))):
            total_cmi.write(str(num_points) + " " + str(cmi))

def initial_mi_vals(west, north, name):
    _, idx_w = discretize_data(west)
    _, idx_n = discretize_data(north)
    mis = cross_mutual_information(idx_w, idx_n, 1, 1)
    mi_string = "%s,%s\n" % (name, str(mis[0]))
    with open("initial_cmis.csv", "a") as initial_cmi:
        initial_cmi.write(mi_string)

def cmi_plot(west, north, name, stepsize=1, stepmax=50):
    max_bits = 10 #define here because it's easy
    _, idx_w = discretize_data(west)
    _, idx_n = discretize_data(north)
    mis = cross_mutual_information(idx_w, idx_n, stepsize, stepmax)
    xmin, xmax, ymin, ymax = 0, stepmax, 0, max_bits
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot(range(0, stepmax, stepsize), mis)
    plt.title(name)
    plt.xlabel("lag")
    plt.ylabel("cmi (bits)")
    plt.savefig("./cmi_plots/" + name)
## phase space methods as needed

def process_num(num_str):
    if num_str == "NA":
        return 0.0
    else:
        return float(num_str)

def check_l2_integrability(data):
    data_abs = np.absolute(data)
    data_sqabs = np.power(data, 2)
    return data_sqabs
    #return np.cumsum(data_sqabs)

"""
In 0-based:
Head 1, 4
Body 0, 7, 10, 13
Arms 2, 5, 3, 6, 15, 14
Legs 8, 9, 11, 12, 16, 17

In 1-based:
Head 2, 5
Body 1, 8, 11, 14
Arms 3, 6, 4, 7, 16, 15
Legs 9, 10, 12, 13, 17, 18

The numbers are same for both north and west
But north is +19

Row 19 in R (18 in Python) is all composed of 0's, same for Row 38 in R (37 in python)
"""

row_dict = {
        0: "Body",
        1: "Head",
        2: "Arms",
        3: "Arms",
        4: "Head",
        5: "Arms",
        6: "Arms",
        7: "Body",
        8: "Legs",
        9: "Legs",
        10: "Body",
        11: "Legs",
        12: "Legs",
        13: "Body",
        14: "Arms",
        15: "Arms",
        16: "Legs",
        17: "Legs"
        }

def processed_glob_series(part_reader, curr_fname):
    part_reader.next() #header
    west = []
    north = []
    for row in part_reader:
        west.append(process_num(row[1]))
        north.append(process_num(row[2]))

    #coherence_over_time(west, north, name=curr_fname)
    #total_amis(west, name=curr_fname + "_west")
    #total_amis(north, name=curr_fname + "_north")
    #total_cmis(west, north, name=curr_fname)
    quick_correlation(west, north)

    #block_phase_coherence(hilbert_phase(west), hilbert_phase(north), name=curr_fname)
    #hilbert_phase_diff_csv(west, north, name=curr_fname)

def filter_nan(member):
    if math.isnan(member):
        return 0
    return member

def process_row(row):
    row = map(float, row)
    return map(filter_nan, row)

def quick_correlation(west, north):
    plt.close()
    #50 seconds worth, meaning 400 frames worth
    cross_corrs = sci_sig.correlate(west, north, "same")
    plt.plot(cross_corrs)
    plt.show()

def unprocessed_glob_series(part_reader, curr_fname):
    rows = list(part_reader)
    for row_idx in xrange(18):
        row_w = process_row(rows[row_idx])
        row_n = process_row(rows[row_idx + 19])
    #poincare_movies(first_row)
        row_fname = curr_fname + ("_row_%02d" % (row_idx,))
        #difference_poincare_ellipse(row_w, row_n, name=row_fname)

if __name__ == "__main__":
    ##### normalize all axes!!!!
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    #unprocessed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*0.csv")
    globs = processed_globs #take this out when necessary
    globs = [globs[0]]
    for curr_path in globs:
        path_splits = os.path.split(curr_path)[1].split(".", 2)
        curr_fname = "".join([path_splits[0], path_splits[1]])
        print curr_fname
        with open(curr_path, "rU") as part_file:
            part_reader = csv.reader(part_file)
            processed_glob_series(part_reader, curr_fname)
