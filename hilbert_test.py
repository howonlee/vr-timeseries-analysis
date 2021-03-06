import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import matplotlib
import math
import operator as op
import plots

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

def block_phase_coherence(first, second, name):
    lambdas = []
    for group in plots.grouper(zip(list(first), list(second)), 10):
        temp_group = filter(lambda x: x, group) #filter out nulls
        xs = np.array(map(op.itemgetter(0), temp_group))
        ys = np.array(map(op.itemgetter(1), temp_group))
        lambdas.append(mean_phase_coherence(xs, ys))
    #for each block, I think
    with open("./phase_lambdas/" + name, "w") as coherence_file:
        coherence_file.write(",".join(map(str, lambdas)))

"""
def total_cmis(west, north, name, num_points=10):
    with open("./cmi_csvs/" + name + ".csv", "w") as total_cmi:
        for idx, cmi in enumerate(cross_mutual_information(west, north, num_points, len(west))):
            total_cmi.write(str(num_points) + " " + str(cmi))
"""

def hilbert_transform_phase_diff(west, north, name):
    #phase diff modulus
    font = {"size": 20}
    matplotlib.rc('font', **font)
    plt.close()
    plt.clf()
    hilbert_w_phase = hilbert_phase(west)
    hilbert_n_phase = hilbert_phase(north)
    diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
    plt.xlabel("time")
    plt.plot(diff)
    plt.ylabel("phase diff(w - n)")
    plt.tight_layout()
    plt.savefig("./phase_diff/" + name)

def hilbert_phase_diff_csv(west, north, name):
    hilbert_w_phase = hilbert_phase(west)
    hilbert_n_phase = hilbert_phase(north)
    diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
    np.savetxt("./phase_diff/" + name, diff, delimiter=",")

def main():
    #should be full phase diff synced, right?
    w = np.sin(np.linspace(0, np.pi * 20, 100000))
    n = np.cos(np.linspace(0, np.pi * 20, 100000))
    hilbert_transform_phase_diff(w, n, "test")
    #print mean_phase_coherence(hilbert_phase(w), hilbert_phase(n))

if __name__ == "__main__":
    #gotta do a good bandpass filter first
    main()
