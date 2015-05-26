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
import scipy.stats.stats as sci_stats

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
        for x in xrange(400, 1500, 10): ########
            west_window = west[0 + (x - 400):x]
            north_window = north[0 + (x - 400):x]
            curr_correlations.append(sci_stats.pearsonr(west_window, north_window)[0])
        plt.plot(curr_correlations, color="blue", alpha=0.1)
        correlations.append(curr_correlations)
    plt.savefig("./wholes/correlations_over_time_mc")
    plt.close()
    average_correlations = np.zeros(len(correlations[0])) #should be 1500
    for correlation in correlations:
        for idx, member in enumerate(correlation):
            if not math.isnan(member):
                average_correlations[idx] += member
    print average_correlations
    average_correlations = np.divide(average_correlations, len(correlations))
    print average_correlations
    print len(correlations)
    plt.plot(average_correlations)
    #got to get the pointwise sd and plot it
    #avg and pointwise sd, I think? need pointwise sd too
    plt.savefig("./wholes/correlations_over_time_summary")

def coherences_over_time(wests, norths):
    #no changes from the original diffs
    plt.close()
    pass
    plt.savefig("./wholes/correlations_over_time")

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
