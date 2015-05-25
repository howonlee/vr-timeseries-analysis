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

def process_num(num_str):
    if num_str == "NA":
        return 0.0
    else:
        return float(num_str)

def processed_glob_series(part_reader, curr_fname):
    part_reader.next() #header
    west = []
    north = []
    for row in part_reader:
        west.append(process_num(row[1]))
        north.append(process_num(row[2]))
    print west

    #coherence_over_time(west, north, name=curr_fname)
    #total_amis(west, name=curr_fname + "_west")
    #total_amis(north, name=curr_fname + "_north")
    #total_cmis(west, north, name=curr_fname)

    #block_phase_coherence(hilbert_phase(west), hilbert_phase(north), name=curr_fname)
    #hilbert_phase_diff_csv(west, north, name=curr_fname)

if __name__ == "__main__":
    processed_globs = glob.glob("/home/curuinor/data/vr_synchrony/*.csv_summed_*.csv")
    globs = processed_globs #take this out when necessary
    #globs = [globs[0]]
    for curr_path in globs:
        path_splits = os.path.split(curr_path)[1].split(".", 2)
        curr_fname = "".join([path_splits[0], path_splits[1]])
        print curr_fname
        with open(curr_path, "rU") as part_file:
            part_reader = csv.reader(part_file)
            processed_glob_series(part_reader, curr_fname)
