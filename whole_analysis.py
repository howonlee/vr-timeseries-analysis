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

def correlations_over_time(wests, norths, name):
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
    plt.savefig("./correlations/" + name)
    plt.close()
    """
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
    """

def coherences_over_time(wests, norths):
    plt.close()
    coherences = []
    for west, north in zip(wests, norths):
        curr_coherences = []
        for x in xrange(0, 100): ######## use np.roll
            curr_coherences.append(plt.cohere(west, north)[0])
        #get freqs and cxy, must plot according to those
        plt.plot(curr_coherences[-1], color="blue", alpha=0.1)
        coherences.append(curr_coherences)
    plt.ylabel("coherence synchrony score")
    plt.xlabel("offset")
    plt.title("simultaneous coherence plot")
    plt.savefig("./wholes/coherence_mc")
    plt.close()
    """
    average_coherences = np.zeros_like(coherences[0][0]) #should be 1500
    for coherence in coherences:
        for idx, member in enumerate(coherence):
            average_coherences += member
    average_coherences = np.divide(average_coherences, len(coherences))
    plt.plot(average_coherences)
    plt.ylabel("coherence synchrony score")
    plt.xlabel("offset")
    plt.title("average coherence plot")
    plt.savefig("./wholes/coherences_over_time_summary")
    """

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

def total_amis(series, name):
    plt.close()
    amis = []
    stepsize = 2
    stepmax = 50
    for member in series:
        ami = auto_mutual_information(member, stepsize, stepmax)
        #get freqs and cxy, must plot according to those
        plt.plot(ami, color="blue", alpha=0.1)
        amis.append(ami)
    plt.ylabel("auto mutual information (bits)")
    plt.xlabel("offset")
    plt.title("simultaneous auto mutual information plot")
    plt.savefig("./wholes/" + name + "_ami_mc")
    plt.close()
    average_amis = np.zeros(len(ami)) #should be 1500
    for ami in amis:
        for idx, member in enumerate(ami):
            average_amis[idx] += member
    average_amis = np.divide(average_amis, len(amis))
    plt.plot(average_amis)
    plt.ylabel("auto mutual information (bits)")
    plt.xlabel("offset")
    plt.title("average auto mutual information plot")
    plt.savefig("./wholes/" + name + "_ami_summary")

def total_cmis(wests, norths):
    plt.close()
    cmis = []
    stepsize = 2
    stepmax = 50
    for west, north in zip(wests, norths):
        cmi = cross_mutual_information(west, north, stepsize, stepmax)
        #get freqs and cxy, must plot according to those
        plt.plot(cmi, color="blue", alpha=0.1)
        cmis.append(cmi)
    plt.ylabel("cross mutual information (bits)")
    plt.xlabel("offset")
    plt.title("simultaneous cross mutual information plot")
    plt.savefig("./wholes/cmi_mc")
    plt.close()
    average_cmis = np.zeros(len(cmi)) #should be 1500
    for cmi in cmis:
        for idx, member in enumerate(cmi):
            average_cmis[idx] += member
    average_cmis = np.divide(average_cmis, len(cmis))
    plt.plot(average_cmis)
    plt.ylabel("cross mutual information (bits)")
    plt.xlabel("offset")
    plt.title("average cross mutual information plot")
    plt.savefig("./wholes/cmi_summary")

def hilbert_phase(data):
    #data to phase, clean and spiffy
    return np.unwrap(np.angle(sci_sig.hilbert(data)))

def mean_phase_coherence(first, second):
    diff = first - second
    mean_sin = np.mean(np.sin(diff))
    mean_cos = np.mean(np.cos(diff))
    #absolute value is not actually ambiguous
    #but it confuses _me_ in complex number land
    #so let's say nix to Euler
    #no more or less complex
    #save this one
    return math.sqrt(mean_sin ** 2 + mean_cos **2)

def total_gammas(wests, norths):
    coherences = []
    for west, north in zip(wests, norths):
        coherences.append(mean_phase_coherence(hilbert_phase(west), hilbert_phase(north)))
    plt.hist(coherences)
    plt.xlabel("gamma")
    plt.ylabel("value")
    plt.savefig("total_gammas")

def total_gamma_hist(wests, norths):
    print "gamma hist"
    coherences = []
    for west, north in zip(wests, norths):
        coherences.append(mean_phase_coherence(hilbert_phase(west), hilbert_phase(north)))
    coherence_max = float(max(coherences))
    coherences = [coherence / coherence_max for coherence in coherences]
    coherences = filter(lambda x: not math.isnan(x), coherences)
    with open("./hist/total_gammas", "w") as gamma_file:
        for coherence in coherences:
            gamma_file.write(str(coherence) + "\n")

def total_cmi_hist(wests, norths):
    print "cmi hist"
    cmis = []
    stepsize = 2
    stepmax = 50
    for west, north in zip(wests, norths):
        cmi = cross_mutual_information(west, north, stepsize, stepmax)[10]
        cmis.append(cmi)
    cmi_max = float(max(cmis))
    cmis = [cmi / cmi_max for cmi in cmis]
    cmis = filter(lambda x: not math.isnan(x), cmis)
    with open("./hist/total_cmis", "w") as cmi_file:
        for cmi in cmis:
            cmi_file.write(str(cmi) + "\n")

def total_corr_hist(wests, norths):
    print "corr hist"
    correlations = []
    for west, north in zip(wests, norths):
        #pearsons correlation, window 400
        correlations.append(sci_stats.stats.pearsonr(west, north)[0])
    correlation_max = float(max(correlations))
    correlations = [correlation / correlation_max for correlation in correlations]
    correlations = filter(lambda x: not math.isnan(x), correlations)
    with open("./hist/total_correlations", "w") as correlation_file:
        for correlation in correlations:
            correlation_file.write(str(correlation) + "\n")

def hilbert_phase(data):
    #data to phase, clean and spiffy
    return np.unwrap(np.angle(sci_sig.hilbert(data)))

def hilbert_phase_diffs(wests, norths):
    plt.close()
    diffs = []
    for west, north in zip(wests, norths):
        hilbert_w_phase = hilbert_phase(west)
        hilbert_n_phase = hilbert_phase(north)
        curr_diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
        plt.plot(curr_diff, color="blue", alpha=0.01)
        diffs.append(curr_diff)
    plt.ylabel("analytic signal difference")
    plt.xlabel("phase")
    plt.title("hilbert phase difference")
    plt.savefig("./wholes/hilbert_mc")
    plt.close()
    average_diffs = np.zeros_like(diffs[0]) #should be 1500
    for diff in diffs:
        for idx, member in enumerate(diff):
            average_diffs[idx] += member
    average_diffs = np.divide(average_diffs, len(diffs))
    plt.plot(average_diffs)
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

    #coherences_over_time(wests, norths)
    #hilbert_phase_diffs(wests, norths)
    #total_amis(wests, "wests")
    #total_amis(norths, "norths")
    #total_gammas(wests, norths)
    total_corr_hist(wests, norths)
    total_gamma_hist(wests, norths)
    total_cmi_hist(wests, norths)

if __name__ == "__main__":
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    globs = processed_globs #take this out when necessary
    #globs = [globs[0]]
    whole_series(globs)
