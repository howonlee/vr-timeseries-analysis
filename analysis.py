import csv
import os.path
import glob
import math
import functools
import itertools
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy.fft as np_f
import numpy.linalg as np_l
import numpy as np
import pandas as pd
import pandas.tools.plotting as pd_plot

def fourier_plot(data):
    plt.clf()
    #not a power spectrum!
    fl_fft = np_f.fft(np.array(data))
    ps = np.abs(fl_fft) ** 2
    fl_freq = np_f.fftfreq(data.shape[-1])
    #plt.semilogy(fl_freq, fl_fft.real, fl_freq, fl_fft.imag)
    plt.semilogy(fl_freq, ps)
    plt.show()

def double_fourier(west, north, name):
    """
    single-sided amplitude spec
    """
    plt.clf()
    west_y = np_f.fft(west)
    north_y = np_f.fft(north)
    freq = np_f.fftfreq(len(west), 1)
    plt.plot(freq, np.abs(west_y), "b")
    plt.plot(freq, np.abs(north), "g")
    plt.title(name)
    plt.xlabel("freq")
    plt.ylabel("|Y(f)|")
    plt.savefig("./fourier_plots/" + name)

def normal_plot(data, name, loglog=False):
    plt.clf()
    if loglog:
        plt.loglog(data)
    else:
        plt.plot(data)
    plt.title(name)
    plt.show()

def double_plot(west, north, name):
    plt.clf()
    plt.plot(west, "b")
    plt.plot(north, "g")
    plt.title(name)
    plt.xlabel("time")
    plt.ylabel("change")
    plt.savefig("./normal_plots/" + name)

def acorr_plot(data):
    plt.clf()
    plt.figure()
    pd_data = pd.Series(data)
    pd_plot.autocorrelation_plot(pd_data)
    plt.show()

def poincare_plot(data, order=1, name="", line=False, sds=False):
    """
    also called a return map
    """
    plt.clf()
    unlagged = data[:-order]
    lagged = np.roll(data, -order)[:-order]
    fig, ax = plt.subplots()
    print sds
    if sds:
        data_mean = np.mean(data)
        #coords = ax.transData.transform(sds)
        ellipse = patches.Ellipse(xy=(data_mean, data_mean), width=(2*sds[1]), height=(2*sds[0]), angle=45, alpha=0.5, color="red")
        ax.add_patch(ellipse)
    if line:
        ax.plot(unlagged, lagged)
    else:
        ax.scatter(unlagged, lagged, s=5)
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.title(name)
    plt.show()

#returns projection of vector u onto the line defined by vector v
def proj(u, v):
    temp = np.dot(u,v) / np.dot(v,v)
    return temp * v

def euclid_norm(x,y=0):
    #what's wrong with the linear algebra norm?
    #something is wrong with it
    return np.sqrt(np.sum((x-y) ** 2))

def ortho_diff(projection, data):
    ortho = []
    for idx, member in enumerate(projection):
        curr_dist = data[idx] - member
        ortho.append(euclid_norm(curr_dist))
    return ortho

def is_within_stddev(data, stddev, mean):
    """
    Check what percentage is within one standard deviation from mean
    You would expect it to follow the 68-95-99.7 rule
    But it doesn't because of strange assumptions they make
    """
    total_num_data = float(len(data))
    within_stddev = 0
    for data_point in data:
        if (data_point > (mean - stddev)) and (data_point < (mean + stddev)):
            within_stddev += 1
    return float(within_stddev) / total_num_data
    #should be like 70%ish?

def ellipse_sds(data, order=1):
    #calculate the standard deviations of the distances to two lines
    #y = x
    #y = -x + (2 * mean)
    data_mean = np.mean(data)
    print data_mean
    data_std = np.std(data)
    unlagged = data[:-order]
    lagged = np.roll(data, -order)[:-order]
    pairs = np.array(zip(unlagged, lagged))
    pairs_list1 = list(pairs) #list of np arrays
    ### gotta shift the entire thing down a little for the subspace
    pairs_list2 = map(lambda x: (x[0], x[1] - (2 * data_mean)), pairs_list1)
    line_1 = np.array([1,1]) #vector defines line, y = x
    line_2 = np.array([1,-1])
    proj_sd1 = map(lambda x: proj(x, line_1), pairs_list1)
    proj_sd2 = map(lambda x: proj(x, line_2), pairs_list2)

    ortho_sd1 = ortho_diff(proj_sd1, pairs_list1)
    ortho_sd2 = ortho_diff(proj_sd2, pairs_list2)
    sd1 = np.std(ortho_sd1)
    sd2 = np.std(ortho_sd2)
    return data_mean, sd1, sd2


def double_poincare(west, north, name, order=1):
    plt.clf()
    unlagged_w = west[:-order]
    lagged_w = np.roll(west, -order)[:-order]
    unlagged_n = north[:-order]
    lagged_n = np.roll(north, -order)[:-order]
    plt.figure()
    plt.scatter(unlagged_w, lagged_w, color="blue", s=3, alpha=0.2)
    plt.scatter(unlagged_n, lagged_n, color="green", s=3, alpha=0.2)
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.title(name)
    plt.savefig("./poincare_plots/" + name)


def difference_poincare(west, north, name, order=1):
    plt.clf()
    west = np.array(west)
    north = np.array(north)
    dts = (west - north)
    unlagged_dts = dts[:-order]
    lagged_dts = np.roll(dts, -order)[:-order]
    plt.figure()
    plt.scatter(unlagged_dts, lagged_dts, color="red", s=3, alpha=0.4)
    #always want to alpha
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.title(name)
    plt.savefig("./difference_poincares/" + name)

def difference_poincare_ellipse(west, north, name, order=1):
    plt.clf()
    plt.close("all")
    west = np.array(west)
    north = np.array(north)
    dts = (west - north)
    fig, ax = plt.subplots()
    unlagged_dts = dts[:-order]
    lagged_dts = np.roll(dts, -order)[:-order]
    data_mean, sd1, sd2 = ellipse_sds(dts)
    ellipse = patches.Ellipse(xy=(data_mean, data_mean), width=(2*sd2), height=(2*sd1), angle=45, alpha=0.5, color="red")
    ax.add_patch(ellipse)
    ax.scatter(unlagged_dts, lagged_dts, s=2)
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.title(name)
    plt.savefig("./difference_poincares_ellipse/" + name)
    with open("./difference_poincares_ellipse/" + name + "_stats", "w") as stats_file:
        stats_file.write("mean:%f\nstd1:%f\nstd1:%f" % (data_mean, sd1, sd2))

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def difference_poincare_movie(west, north, order=1, name="", points_per_frame=100):
    #add in the things frame by frame
    total_max = max([max(west), max(north)])
    plt.clf()
    plt.close("all")

    fig, ax = plt.subplots()
    ax.axis([0, total_max, 0, total_max])
    ax.set_autoscale_on(False)
    #set the damned axes
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.title(name)

    west = np.array(west)
    north = np.array(north)
    dts = (west - north)

    unlagged_dts = dts[:-order]
    lagged_dts = np.roll(dts, -order)[:-order]
    to_plot = zip(unlagged_dts, lagged_dts)
    print len(to_plot)
    curr_plot = 0
    for group in grouper(to_plot, points_per_frame):
        group = filter(lambda x: x, group)
        print len(group)
        xs = map(operator.itemgetter(0), group)
        ys = map(operator.itemgetter(1), group)
        ax.scatter(xs, ys, alpha=0.3)
        ax.axis([0, total_max, 0, total_max])
        plt.savefig("./difference_poincare_movies/" + name + ("_%02d" % (curr_plot,)))
        curr_plot += 1


def correlation_over_time(west, north, name, order=1, points_per_frame=100):
    plt.clf()
    plt.close("all")
    wmin, wmax, nmin, nmax = min(west), max(west), min(north), max(north)
    fig, ax = plt.subplots()
    ax.axis([wmin, wmax, nmin, nmax])
    ax.set_autoscale_on(False)
    plt.xlabel("west")
    plt.ylabel("north")
    plt.title(name)
    west = np.array(west)
    north = np.array(north)
    to_plot = zip(west, north)
    curr_plot = 0
    for group in grouper(to_plot, points_per_frame):
        group = filter(lambda x: x, group)
        xs = map(operator.itemgetter(0), group)
        ys = map(operator.itemgetter(1), group)
        data_mean, sd1, sd2 = ellipse_sds(group)
        ellipse = patches.Ellipse(xy=(data_mean, data_mean), width=(2*sd2), height=(2*sd1), angle=45, alpha=0.5, color="red")
        ax.add_patch(ellipse)
        ax.scatter(xs, ys, alpha=0.3)
        ax.axis([wmin, wmax, nmin, nmax])
        plt.savefig("./correlation_movies/" + name + ("_%02d" % (curr_plot,)))
        curr_plot += 1
        with open("./correlation_movies/" + name + "_%02d_stats" % (curr_plot,), "w") as stats_file:
            stats_file.write("mean:%f\nstd1:%f\nstd1:%f" % (data_mean, sd1, sd2))

def recurrence_plot(data):
    num_pts = data.size
    ret_mat = np.zeros((num_pts, num_pts))
    for x in xrange(num_pts):
        for y in xrange(num_pts):
            ret_mat[x,y] = abs(data[x] - data[y])
    plt.matshow(ret_mat)
    plt.show()

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
    data1, data2 = list(data1), list(data2)
    freq1, freq2 = probdict(data1), probdict(data2)
    prob_hist, x_edges, y_edges = np.histogram2d(data1, data2, normed=True)
    prob_hist = prob_hist.ravel()[np.nonzero(prob_hist.ravel())]
    log_prob_hist = np.log2(prob_hist)
    entropy = 0
    for x in xrange(prob_hist.shape[0]):
        entropy -= prob_hist[x] * log_prob_hist[x]
    return entropy


def arnold_tongue(data):
    pass

def fractal_dimension(data):
    pass

def lyapunov_exponent(data):
    pass

def phase_space_embedding(data):
    #logical prerequisite to the others
    pass

def knn_fit(data):
    pass

def neural_net_fit(data):
    pass

def mutual_information_plot(data):
    pass

def lag_plot(data):
    """
    minimum of autocorrelation heuristic says:
    34
    actually, we should use the mi for this in actuality
    """
    tau = 34
    unlagged = data[:-tau]
    lagged = np.roll(data, -tau)[:-tau]
    plt.figure()
    plt.plot(unlagged, "b")
    plt.plot(lagged, "g")
    plt.show()

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
    #difference_poincare_ellipse(west, north, name=curr_fname)
    #difference_poincare_movie(west, north, name=curr_fname)
    correlation_over_time(west, north, name=curr_fname)

def filter_nan(member):
    if math.isnan(member):
        return 0
    return member

def process_row(row):
    row = map(float, row)
    return map(filter_nan, row)

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
    print entropy([1,2,3,4,65,3,2,5,32,4,3,5])
    """
    for curr_path in globs:
        path_splits = os.path.split(curr_path)[1].split(".", 2)
        curr_fname = "".join([path_splits[0], path_splits[1]])
        print curr_fname
        with open(curr_path, "rU") as part_file:
            part_reader = csv.reader(part_file)
            processed_glob_series(part_reader, curr_fname)
    """
