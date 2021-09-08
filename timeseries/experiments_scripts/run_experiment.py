import os
import sys
import torch
import random
import argparse
import numpy as np
import scipy

sys.path.append('../../timeseries/')
sys.path.append('../../src/')

import factors
import state_models
import observation_models
from ssm_em import StateSpaceEM
from nonlinear_ssm import NonLinearStateSpace_EM

from scipy.stats import norm
from scipy.stats import zscore
from ssm_em import StateSpaceEM
from sklearn.covariance import EmpiricalCovariance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF

import darts
from darts.models import TCNModel
from darts.models import GaussianProcessFilter
from darts.utils.likelihood_models import GaussianLikelihoodModel
from darts.timeseries import TimeSeries
from exp_utils import load_sunspots, load_energy, load_synthetic, load_airfoil, load_sunspots_e1

'''
sys.path.append('../../timeseries/kalman-jax-master')
from jax.experimental import optimizers
#from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot
'''
def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_linear_SSM(x_tr, **kwargs):

    dx = x_tr.shape[1]
    sm = state_models.LinearStateModel(args.dz)
    om = observation_models.LinearObservationModel(dx, args.dz, noise_x=1.)
    
    if args.init_w_pca == 1:
        om.pca_init(x_tr)
        
    ssm_em_lin = StateSpaceEM(x_tr, observation_model=om, state_model=sm)
    ssm_em_lin.run()
    
    return ssm_em_lin


def train_linear_hsk_SSM(x_tr, **kwargs):

    dx = x_tr.shape[1]
    sm_hs = state_models.LinearStateModel(args.dz)
    om_hs = observation_models.HCCovObservationModel(dx, args.dz, args.du)
    if args.init_w_pca == 1:
        om_hs.pca_init(x_tr)
    hs_model = StateSpaceEM(x_tr, observation_model=om_hs, state_model=sm_hs)
    hs_model.run()
    
    return hs_model
    
    
def train_nonlinear_SSM(x_tr, **kwargs):
    '''
    to be updated; currently doesn't run
    LSEMStateModel -> sm_hs = state_models.LSEMStateModel(args.dz,args.dk# + param for basis func)
    '''
    nonlin_model = NonLinearStateSpace_EM(x_tr,args.dz, args.dk)
    nonlin_model.run()
    
    return nonlin_model


class TCNModel_ext(TCNModel):
    def __init__(self, x_tr):
            self.pred_mean = 0
            self.pred_var = 0
            self.tcnmodel = TCNModel(
                                    input_chunk_length=args.in_len,
                                    output_chunk_length=args.out_len,
                                    kernel_size=args.kernel_size,
                                    num_filters=args.num_filter,
                                    dilation_base=args.d_base,
                                    dropout=args.dropout,
                                    random_state=args.seed,
                                    n_epochs=args.epochs,
                                    likelihood=GaussianLikelihoodModel())
            
            self.tcnmodel.fit(x_tr)
    
    def predict(self, x_te):
        
        backtest_en = self.tcnmodel.historical_forecasts(
                                            series=x_te,
                                            num_samples=50,
                                            start=4,
                                            forecast_horizon=1,
                                            retrain=False,
                                            verbose=True)
        tcn_sigma = (backtest_en.quantile_timeseries(quantile=0.975) - \
                     backtest_en.quantile_timeseries(quantile=0.025))/2
        
        pred_data = np.mean(backtest_en._xa, 2)
        muVector = np.mean(pred_data, axis=0)
        cov = EmpiricalCovariance().fit(pred_data)
        Sigma = cov.covariance_
        self.muVector = muVector
        self.Sigma = Sigma
        
        return 0, backtest_en, tcn_sigma

    
    def compute_predictive_log_likelihood(self, x_te):
        test_data = x_te.all_values().squeeze()
        if test_data.shape[1] > 1:
            return scipy.stats.multivariate_normal.logpdf(test_data, 
                                                          self.muVector, self.Sigma).sum()
        else:
            return scipy.stats.norm.logpdf(test_data, numpy.mean(test_data), 
                                           numpy.var(test_data)).sum()

        
def train_deep_tcn(x_tr):
    
    deep_tcn = TCNModel_ext(x_tr)
    
    return deep_tcn

class GaussianProcessFilter_ext(GaussianProcessFilter):
    def __init__(self, x_tr):
        kernel = RBF(length_scale=args.gp_kernel_width)
        self.gp_obj = GaussianProcessFilter(kernel,  alpha=args.gp_noise_dist/2, n_restarts_optimizer=50)

        filtered_x = self.gp_obj.filter(x_tr, num_samples=100)
        
        pred_data = np.mean(filtered_x._xa, 2)
        muVector = np.mean(pred_data, axis=0)
        cov = EmpiricalCovariance().fit(pred_data)
        Sigma = cov.covariance_
        self.muVector = muVector
        self.Sigma = Sigma
        
    def predict(self, x_te):
        
        return 0, self.gp_obj.filter(x_te, num_samples=1), 0
    
    def compute_predictive_log_likelihood(self, x_te):
        test_data = x_te.all_values().squeeze()
        if test_data.shape[1] > 1:
            return scipy.stats.multivariate_normal.logpdf(test_data, 
                                                          self.muVector, self.Sigma).sum()
        else:
            return scipy.stats.norm.logpdf(test_data, numpy.mean(test_data), 
                                           numpy.var(test_data)).sum()


def train_gp(x_tr):

    gp_obj = GaussianProcessFilter_ext(x_tr)

    return gp_obj


def train_hkalmal(x_tr):

    gp_obj = GaussianProcessFilter_ext(x_tr)

    return gp_obj

def compute_mape(s_true, s_pred):
    
    if isinstance(s_true, darts.timeseries.TimeSeries):
        res = darts.metrics.mape(s_pred, s_true) 
    else:
        res = mape(s_true, s_pred)
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="airfoil")
    parser.add_argument('--model_name', type=str, default="lin_ssm")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--whiten', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--dz', type=int, default=2)
    parser.add_argument('--du', type=int, default=2)
    parser.add_argument('--dk', type=int, default=2)
    parser.add_argument('--init_with_pca', type=int, default=0)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--ts', type=int, default=0)
    parser.add_argument('--in_len', type=int, default=3)
    parser.add_argument('--out_len', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--num_filter', type=int, default=4)
    parser.add_argument('--d_base', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--init_w_pca', type=int, default=0)
    parser.add_argument('--results_file', type=str, default='first_results.txt')
    parser.add_argument('--gp_kernel_width', type=float, default='0.001')
    parser.add_argument('--gp_noise_dist', type=float, default='0.004')
    args = parser.parse_args()

    reset_seeds(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # load data
    if args.dataset == 'sunspots':
        x_tr, x_va, x_te, s_tr_x = load_sunspots_e1(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'energy':
        x_tr, x_va, x_te, s_tr_x = load_energy(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'synthetic':
        x_tr, x_va, x_te, s_tr_x = load_synthetic(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'airfoil':
        x_tr, x_va, x_te, s_tr_x = load_airfoil(ts=args.ts, train_ratio=0.5)
  

    # train model
    if args.model_name == 'linear_hsk_ssm':
        model = 'linear_hsk_SSM'
    if args.model_name == 'lin_ssm':
        model = 'linear_SSM'
    if args.model_name == 'nonlinear_ssm':
        model = 'nonlinear_SSM'
    if args.model_name == 'deep_tcn':
        model = 'deep_tcn'
    if args.model_name == 'gp':
        model =  'gp'
    
    print(x_tr)
    trained_model = eval('train_' + model)(x_tr)
        
    # make predictions
    _, mu_pred_x_tr, sigma_pred_x_tr = trained_model.predict(x_tr)
    _, mu_pred_x_te, sigma_pred_x_te = trained_model.predict(x_te)
    _, mu_pred_x_va, sigma_pred_x_va = trained_model.predict(x_va)
    
    # for debug; the folowing two had same values for ssm models
    # print(mu_pred_x_tr.shape)
    # print(x_tr.shape)
   
    # compute metrics
    mape_tr = compute_mape(x_tr, mu_pred_x_tr)
    mape_va = compute_mape(x_va, mu_pred_x_va)
    mape_te = compute_mape(x_te, mu_pred_x_te)

    pll_tr = trained_model.compute_predictive_log_likelihood(x_tr)
    pll_va = trained_model.compute_predictive_log_likelihood(x_va)
    pll_te = trained_model.compute_predictive_log_likelihood(x_te)
    
    # percentage of captured points
    # capture_tr = (p_low_tr.lt(y_tr) * y_tr.lt(p_high_tr)).float().mean()
    # capture_va = (p_low_va.lt(y_va) * y_va.lt(p_high_va)).float().mean()
    # capture_te = (p_low_te.lt(y_te) * y_te.lt(p_high_te)).float().mean()
    
    # print and store results
    print("{:<22} | {:<22} | {:.5f} {:.5f} {:.5f} |{:.5f} {:.5f} {:.5f} | {:<2} # {}".format(
            args.model_name, args.dataset,
            pll_tr, pll_va, pll_te,
            mape_tr, mape_va, mape_te,
            args.seed, 
            str(vars(args))))
    
    
    text_file = open("./results/{}".format(args.results_file), "a+")
    text_file.write("{:<22} | {:<22} | {:.5f} {:.5f} {:.5f} |{:.5f} {:.5f} {:.5f} | {:<2} # {}".format(
            args.model_name, args.dataset,
            pll_tr, pll_va, pll_te,
            mape_tr, mape_va, mape_te,
            args.seed, 
            str(vars(args))))
    text_file.write('\n') 
    text_file.close()