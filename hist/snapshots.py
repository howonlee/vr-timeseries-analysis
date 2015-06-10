import scipy.stats as sci_stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

with open("total_gammas") as gamma_file:
    gammas = map(lambda x: float(x.strip()), list(gamma_file))
    #gammas_sum = float(sum(gammas))
    #norm_gammas = [gamma / gammas_sum for gamma in gammas]

with open("total_cmis") as cmi_file:
    cmis = map(lambda x: float(x.strip()), list(cmi_file))
    #cmis_sum = float(sum(cmis))
    #norm_cmis = [cmi / cmis_sum for cmi in cmis]

plt.hist(gammas)
plt.xlabel("gamma")
plt.ylabel("number of pairs")
plt.tight_layout()
plt.savefig("gamma_snapshot")
