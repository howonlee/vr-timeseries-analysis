import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt

def hilbert_phase(transformed):
    hilbert_real, hilbert_imag = np.real(transformed), np.imag(transformed)
    return np.arctan2(hilbert_imag, hilbert_real)

def hilbert_transform_phase_diff(west, north, name):
    #phase diff modulus
    hilbert_w_phase = hilbert_phase(sci_sig.hilbert(west))
    hilbert_n_phase = hilbert_phase(sci_sig.hilbert(north))
    diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
    plt.title(name)
    plt.xlabel("time")
    plt.plot(diff)
    plt.ylabel("phase diff(w - n)")
    plt.savefig("./phase_diff/" + name)

def hilbert_phase_coherence(west, north):
    #calculate the index of synchronization
    hilbert_w_phase = hilbert_phase(sci_sig.hilbert(west))
    hilbert_n_phase = hilbert_phase(sci_sig.hilbert(north))
    diff = np.exp(1j * (hilbert_w_phase - hilbert_n_phase))
    return np.absolute(np.mean(diff))

def main():
    #should be full phase diff synced, right?
    w = np.random.random(100000)
    n = np.random.random(100000)
    hilbert_transform_phase_diff(w, n, "test")
    print hilbert_phase_coherence(w, n)

if __name__ == "__main__":
    main()
