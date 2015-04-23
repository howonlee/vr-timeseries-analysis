import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import math

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
    return math.sqrt(mean_sin ** 2 + mean_cos **2)

def hilbert_transform_phase_diff(west, north, name):
    #phase diff modulus
    hilbert_w_phase = hilbert_phase(west)
    hilbert_n_phase = hilbert_phase(north)
    diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
    plt.title(name)
    plt.xlabel("time")
    plt.plot(diff)
    plt.ylabel("phase diff(w - n)")
    plt.show()
    #plt.savefig("./phase_diff/" + name)

def main():
    #should be full phase diff synced, right?
    w = np.sin(np.linspace(0, np.pi * 20, 100000))
    n = np.cos(np.linspace(0, np.pi * 20, 100000))
    hilbert_transform_phase_diff(w, n, "test")
    #print mean_phase_coherence(hilbert_phase(w), hilbert_phase(n))

if __name__ == "__main__":
    #gotta do a good bandpass filter first
    main()
