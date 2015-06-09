import scipy.stats as sci_stats
import numpy as np
import matplotlib.pyplot as plt

with open("total_correlations") as correlations_file:
    corrs = map(lambda x: float(x.strip()), list(correlations_file))
    #corrs_sum = float(sum(corrs))
    #norm_corrs = [corr / corrs_sum for corr in corrs]

with open("total_gammas") as gamma_file:
    gammas = map(lambda x: float(x.strip()), list(gamma_file))
    #gammas_sum = float(sum(gammas))
    #norm_gammas = [gamma / gammas_sum for gamma in gammas]

with open("total_cmis") as cmi_file:
    cmis = map(lambda x: float(x.strip()), list(cmi_file))
    #cmis_sum = float(sum(cmis))
    #norm_cmis = [cmi / cmis_sum for cmi in cmis]

print sci_stats.ks_2samp(corrs, gammas)
