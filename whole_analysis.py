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
import scipy.stats as sci_stats

def process_num(num_str):
    if num_str == "NA":
        return 0.0
    else:
        return float(num_str)

def correlations_over_time(wests, norths):
    plt.close()
    correlations = []
    for west, north in zip(wests, norths):
        #pearsons correlation, window 400
        curr_correlations = []
        for x in xrange(0, 100): ########
            west_window = west[0 : 400]
            north_window = north[x : 400+x]
            curr_correlations.append(sci_stats.stats.pearsonr(west_window, north_window)[0])
        plt.plot(curr_correlations, color="blue", alpha=0.1)
        correlations.append(curr_correlations)
    plt.ylabel("correlation synchrony score")
    plt.xlabel("offset")
    plt.title("simultaneous correlation plot")
    plt.savefig("./wholes/correlations_over_time_mc")
    plt.close()
    average_correlations = np.zeros(len(correlations[0])) #should be 1500
    for correlation in correlations:
        for idx, member in enumerate(correlation):
            if not math.isnan(member):
                average_correlations[idx] += member
    average_correlations = np.divide(average_correlations, len(correlations))
    #not actually a correlation matrix, but a matrix with a bunch of correlations in it
    #confusing, no?
    #corr_mat = np.zeros((len(correlations), len(correlations[0])))
    #for x, correlation in enumerate(correlations):
    #    for y, member in enumerate(correlation):
    #        if not math.isnan(member):
    #            corr_mat[x,y] = member
    #stds = np.std(corr_mat, axis=0)
    #dfs = np.ones_like(stds) * 6000
    #plt.errorbar(range(average_correlations.size), average_correlations, yerr=sci_stats.t.ppf(0.95, dfs) * stds)
    plt.plot(average_correlations)
    plt.ylabel("correlation synchrony score")
    plt.xlabel("offset")
    plt.title("average correlation plot")
    plt.savefig("./wholes/correlations_over_time_summary")

def coherences_over_time(wests, norths):
    plt.close()
    coherences = []
    for west, north in zip(wests, norths):
        curr_coherences = []
        for x in xrange(0, 100): ######## use np.roll
####################################################
####################################################
####################################################
            west_window = west[0 : 400]
            north_window = north[x : 400+x]
            curr_coherences.append(sci_sig.coherence(west_window, north_window))
        #get freqs and cxy, must plot according to those
        plt.plot(curr_coherences, color="blue", alpha=0.1)
        coherences.append(curr_coherences)
    plt.ylabel("coherence synchrony score")
    plt.xlabel("offset")
    plt.title("simultaneous coherence plot")
    plt.savefig("./wholes/coherence_mc")
    plt.close()
    average_coherences = np.zeros(len(coherences[0])) #should be 1500
    for coherence in coherences:
        for idx, member in enumerate(coherence):
            if not math.isnan(member):
                average_coherences[idx] += member
    average_coherences = np.divide(average_coherences, len(coherences))
    plt.plot(average_coherences)
    plt.ylabel("coherence synchrony score")
    plt.xlabel("offset")
    plt.title("average coherence plot")
    plt.savefig("./wholes/coherences_over_time_summary")

def total_amis(wests, norths):
    plt.close()
    #abstract away the ami information
    pass
    plt.savefig("./wholes/amis")

def total_cmis(wests, norths):
    plt.close()
    pass
    plt.savefig("./wholes/cmis")

def hilbert_phase_diffs(wests, norths):
    plt.close()
    #no changes from the original diffs
    pass
    plt.savefig("./wholes/hilbert_phase_diffs")

def block_phase_coherences(wests, norths):
    plt.close()
    #no changes from the original diffs
    pass
    plt.savefig("./wholes/block_phase_coherences")

def whole_series(globs):
    wests = []
    norths = []
    for curr_path in globs:
        path_splits = os.path.split(curr_path)[1].split(".", 2)
        curr_fname = "".join([path_splits[0], path_splits[1]])
        print curr_fname
        with open(curr_path, "rU") as part_file:
            part_reader = csv.reader(part_file)
            part_reader.next()
            curr_west = []
            curr_north = []
            for row in part_reader:
                curr_west.append(process_num(row[1]))
                curr_north.append(process_num(row[2]))
            wests.append(curr_west)
            norths.append(curr_north)

    correlations_over_time(wests, norths)
    #coherences_over_time(wests, norths)

if __name__ == "__main__":
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    globs = processed_globs #take this out when necessary
    #globs = [globs[0]]
    whole_series(globs)
