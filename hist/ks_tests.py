import scipy.stats as sci_stats
import numpy as np

buckets = np.linspace(0, 1, 30)
print buckets

with open("total_correlations") as correlations_file:
    print map(lambda x: int(x.strip()), list(correlations_file))

with open("total_gammas") as gamma_file:
    print map(lambda x: int(x.strip()), list(gamma_file))

with open("total_cmis") as cmi_file:
    print map(lambda x: int(x.strip()), list(cmi_file))
