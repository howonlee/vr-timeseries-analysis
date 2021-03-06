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
import matplotlib
import numpy.fft as np_f
import numpy.linalg as np_l
import numpy as np
import pandas as pd
import pandas.tools.plotting as pd_plot
import scipy.signal as sci_sig

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
    font = {"size": 22}
    matplotlib.rc("font", **font)
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
    plt.tight_layout()
    plt.savefig("./poincare_plots/" + name)

#returns pkrojection of vector u onto the line defined by vector v
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
    plt.close()
    font = {"size": 22}
    matplotlib.rc("font", **font)
    unlagged_w = west[:-order]
    lagged_w = np.roll(west, -order)[:-order]
    unlagged_n = north[:-order]
    lagged_n = np.roll(north, -order)[:-order]
    plt.figure()
    plt.scatter(unlagged_w, lagged_w, color="blue", s=3, alpha=0.2)
    plt.scatter(unlagged_n, lagged_n, color="green", s=3, alpha=0.2)
    plt.tight_layout()
    plt.xlabel("unlagged")
    plt.ylabel("lagged")
    plt.savefig("./poincare_plots/" + name)


def difference_poincare(west, north, name, order=1):
    plt.clf()
    font = {"size": 22}
    matplotlib.rc("font", **font)
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
    plt.tight_layout()
    plt.savefig("./poincare_plots/" + name)

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

#tad useless here
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

