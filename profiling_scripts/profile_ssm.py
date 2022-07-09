import os
import sys
import random
import argparse
import numpy as np
import scipy
from matplotlib import pyplot
import numpy

sys.path.append("../timeseries/")
sys.path.append("../timeseries/experiments_scripts/")
sys.path.append("../src/")

import factor
import state_models
import observation_models
from ssm_em import StateSpaceEM
from nonlinear_ssm import NonLinearStateSpace_EM
from exp_utils import load_synthetic_e1


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


dz = 2
seed = 0
reset_seeds(seed)
x_tr, x_va, x_te, x_te_na, s_tr_x = load_synthetic_e1(train_ratio=0.75)
x_tr = x_tr[:1000]
dx = x_tr.shape[1]
sm = state_models.LinearStateModel(dz)
om = observation_models.LinearObservationModel(dx, dz, noise_x=1.0)
# om.pca_init(x_tr, smooth_window=20)

ssm_em_lin = StateSpaceEM(x_tr, observation_model=om, state_model=sm)
ssm_em_lin.run()
