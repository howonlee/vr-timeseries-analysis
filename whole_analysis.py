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
            curr_coherences.append(plt.cohere(west, north)[0])
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

def total_cmis(wests, norths):
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

def hilbert_phase(data):
    #data to phase, clean and spiffy
    return np.unwrap(np.angle(sci_sig.hilbert(data)))

def hilbert_phase_diffs(wests, norths):
    plt.close()
    diffs = []
    for west, north in zip(wests, norths):
        curr_diffs = []
        for x in xrange(0, 100):
            hilbert_w_phase = hilbert_phase(west)
            hilbert_n_phase = hilbert_phase(north)
            diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
            curr_diffs.append(diff)
        #get freqs and cxy, must plot according to those
        plt.plot(curr_diffs, color="blue", alpha=0.1)
        diffs.append(curr_diffs)
    plt.ylabel("analytic signal difference")
    plt.xlabel("phase")
    plt.title("hilbert phase difference")
    plt.savefig("./wholes/hilbert_mc")
    plt.close()
    average_diffs = np.zeros(len(diffs[0])) #should be 1500
    for diff in diffs:
        for idx, member in enumerate(diff):
            if not math.isnan(member):
                average_diff[idx] += member
    average_diffs = np.divide(average_diffs, len(diffs))
    plt.plot(avarage_diffs)
    plt.ylabel("analytic signal difference")
    plt.xlabel("phase")
    plt.title("average hilbert phase difference")
    plt.savefig("./wholes/hilbert_summary")

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

    #correlations_over_time(wests, norths)
    #coherences_over_time(wests, norths)
    #hilbert_phase_diffs(wests, norths)
    total_amis(wests, norths)
    total_cmis(wests, norths)

if __name__ == "__main__":
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    globs = processed_globs #take this out when necessary
    #globs = [globs[0]]
    whole_series(globs)
