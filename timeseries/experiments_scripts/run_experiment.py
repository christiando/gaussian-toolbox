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
from exp_utils import *

import newt
import objax

'''
sys.path.append('../../timeseries/kalman-jax-master')
from jax.experimental import optimizers
#from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot
'''

class PredictiveDensity:
    def __init__(self, mu, sigma):
        if mu.ndim == 1:
            self.mu = np.array([mu]).T
        else:
            self.mu = np.array(mu)
        if sigma.ndim == 1:
            self.Sigma = np.array([sigma]).T
        else:    
            self.Sigma = np.array(sigma)
        
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


class jax_HSK_model(object):
    def __init__(self, x_tr):
        self.x_tr = x_tr
        self.t_tr = np.array([np.arange(x_tr.shape[0])]).T
        self.inf_args = {
            "power": 0.5,  # the EP power
        }
        self.model = self._train()

        
    def _train(self):
        X = self.t_tr
        Y = self.x_tr
        N = X.shape[0]
        batch_size = N  # 100

        var_f1 = 3.  # GP variance
        len_f1 = 10.  # GP lengthscale
        var_f2 = 1.  # GP variance
        len_f2 = 1.  # GP lengthscale

        kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, kern2])
        lik = newt.likelihoods.HeteroscedasticNoise()
        #model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        model = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        lr_adam = 0.01
        lr_newton = 0.05
        iters = 200
        opt_hypers = objax.optimizer.Adam(model.vars())
        energy = objax.GradValues(model.energy, model.vars())


        @objax.Function.with_vars(model.vars() + opt_hypers.vars())
        def train_op():
            model.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

        train_op = objax.Jit(train_op)

        for i in range(1, iters + 1):
            loss = train_op()
        return model
    
    def compute_predictive_log_likelihood(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        model_te = self._train_test_model(x_te, t_te)
        posterior_mean, posterior_var = model_te.predict(X=t_te)
        #link = model_te.likelihood.link_fn
        #mean_te, std_te = posterior_mean[:, 0], np.sqrt(posterior_var[:, 0] + link(posterior_mean[:, 1]) ** 2)
        #llk = - .5 * np.sum(((x_te-mean_te) / std_te) ** 2 + np.log(2 * np.pi * std_te ** 2))
        return float(model_te.compute_log_lik())
    
    def compute_predictive_density(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        x_te_nan_idx = np.where([np.any(np.isnan(x_te), axis=1)])[0]
        x_te_not_nan_idx = np.where([np.logical_not(np.any(np.isnan(x_te), axis=1))])[0]
        model_te = self._train_test_model(x_te[x_te_not_nan_idx], t_te[x_te_not_nan_idx])
        posterior_mean, posterior_var = model_te.predict(X=t_te)
        link = model_te.likelihood.link_fn
        mean_te, std_te = posterior_mean[:, 0], np.sqrt(posterior_var[:, 0] + link(posterior_mean[:, 1]) ** 2)
        return PredictiveDensity(mean_te, std_te)
   
    def _train_test_model(self, x_te, t_te):
        var_f1 = self.model.kernel.kernel0.variance  # GP variance
        len_f1 = self.model.kernel.kernel0.lengthscale  # GP lengthscale
        var_f2 = self.model.kernel.kernel1.variance  # GP variance
        len_f2 = self.model.kernel.kernel1.lengthscale  # GP lengthscale

        kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, kern2])
        lik = newt.likelihoods.HeteroscedasticNoise()
        t_te = np.array([np.arange(x_te.shape[0])]).T
        #model_te = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        model_te = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        lr_newton = 0.05
        for i in range(50):
            model_te.inference(lr=lr_newton, **self.inf_args)
        return model_te
    
def train_newt_hsk(x_tr, **kwargs):
    
    jax_hsk_model = jax_HSK_model(x_tr)
    return jax_hsk_model


class jax_Gaussian_model(object):
    def __init__(self, x_tr):
        self.x_tr = x_tr
        self.t_tr = np.array([np.arange(x_tr.shape[0])]).T
        self.inf_args = {
            "power": 0.5,  # the EP power
        }
        self.model = self._train()

        
    def _train(self):
        X = self.t_tr
        Y = self.x_tr
        N = X.shape[0]
        batch_size = N  # 100

        var_f1 = 3.  # GP variance
        len_f1 = 10.  # GP lengthscale
        #var_f2 = 1.  # GP variance
        #len_f2 = 1.  # GP lengthscale

        kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        #kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, ])
        #lik = newt.likelihoods.HeteroscedasticNoise()
        lik = newt.likelihoods.Gaussian()
        #model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        model = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)

        lr_adam = 0.01
        lr_newton = 0.05
        iters = 200
        opt_hypers = objax.optimizer.Adam(model.vars())
        energy = objax.GradValues(model.energy, model.vars())


        @objax.Function.with_vars(model.vars() + opt_hypers.vars())
        def train_op():
            model.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

        train_op = objax.Jit(train_op)

        for i in range(1, iters + 1):
            loss = train_op()
        return model
    
    def compute_predictive_log_likelihood(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        model_te = self._train_test_model(x_te, t_te)
        return float(model_te.compute_log_lik())
    
    def compute_predictive_density(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        x_te_nan_idx = np.where([np.any(np.isnan(x_te), axis=1)])[0]
        x_te_not_nan_idx = np.where([np.logical_not(np.any(np.isnan(x_te), axis=1))])[0]
        model_te = self._train_test_model(x_te[x_te_not_nan_idx], t_te[x_te_not_nan_idx])
        mean_te, std_te = model_te.predict_y(t_te)
        return PredictiveDensity(mean_te, std_te)
    
    def _train_test_model(self, x_te, t_te):
        var_f1 = self.model.kernel.kernel0.variance  # GP variance
        len_f1 = self.model.kernel.kernel0.lengthscale  # GP lengthscale
        #var_f2 = self.model.kernel.kernel1.variance  # GP variance
        #len_f2 = self.model.kernel.kernel1.lengthscale  # GP lengthscale

        kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        #kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, ])
        #lik = newt.likelihoods.HeteroscedasticNoise()
        lik = newt.likelihoods.Gaussian()
        lik.transformed_variance =  self.model.likelihood.transformed_variance
        t_te = np.array([np.arange(x_te.shape[0])]).T
        #model_te = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        model_te = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        lr_newton = 0.05
        for i in range(50):
            model_te.inference(lr=lr_newton, **self.inf_args)
        return model_te
    
def train_newt_gauss(x_tr, **kwargs):
    
    jax_gauss_model = jax_Gaussian_model(x_tr)
    return jax_gauss_model


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
    parser.add_argument('--dataset', type=str, default="sunspots")
    parser.add_argument('--model_name', type=str, default="lin_ssm")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--whiten', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--dz', type=int, default=2)
    parser.add_argument('--du', type=int, default=1)
    parser.add_argument('--dk', type=int, default=1)
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
    parser.add_argument('--exp_num', type=str, default="1")
    args = parser.parse_args()

    reset_seeds(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # load data
    if args.dataset == 'sunspots':
        x_tr, x_va, x_te, x_te_na, s_tr_x = eval('load_sunspots_e' + args.exp_num)(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'energy':
        x_tr, x_va, x_te, x_te_na, s_tr_x = eval('load_energy_e' + args.exp_num)(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'synthetic':
        x_tr, x_va, x_te, x_te_na, s_tr_x = eval('load_synthetic_e' + args.exp_num)(ts=args.ts, train_ratio=0.5)
    if args.dataset == 'airfoil':
        x_tr, x_va, x_te, x_te_na, s_tr_x = eval('load_airfoil_e' + args.exp_num)(ts=args.ts, train_ratio=0.5)
  

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
    if args.model_name == 'newt_gauss':
        model = 'newt_gauss'
    if args.model_name == 'newt_hsk':
        model = 'newt_hsk'
    trained_model = eval('train_' + model)(x_tr)
        
    '''
    # make predictions
    _, mu_pred_x_tr, sigma_pred_x_tr = trained_model.predict(x_tr,smoothed=True)
    _, mu_pred_x_te, sigma_pred_x_te = trained_model.predict(x_te,smoothed=True)
    _, mu_pred_x_va, sigma_pred_x_va = trained_model.predict(x_va,smoothed=True)
    '''
    
    pred_x_tr = trained_model.compute_predictive_density(x_tr)
    pred_x_va = trained_model.compute_predictive_density(x_va)
    pred_x_te = trained_model.compute_predictive_density(x_te)
    
    mu_pred_x_tr = pred_x_tr.mu
    mu_pred_x_va = pred_x_va.mu
    mu_pred_x_te = pred_x_te.mu
    
    sigma_pred_x_tr = pred_x_tr.Sigma
    sigma_pred_x_va = pred_x_va.Sigma
    sigma_pred_x_te = pred_x_te.Sigma
    print(mu_pred_x_tr.shape, x_tr.shape)
    # compute metrics
    mape_tr = compute_mape(x_tr, mu_pred_x_tr[:,0])
    mape_va = compute_mape(x_va, mu_pred_x_va[:,0])
    mape_te = compute_mape(x_te, mu_pred_x_te[:,0])

    pll_tr = trained_model.compute_predictive_log_likelihood(x_tr)
    pll_va = trained_model.compute_predictive_log_likelihood(x_va)
    pll_te = trained_model.compute_predictive_log_likelihood(x_te)
    
    capture_tr_all_x = []
    capture_va_all_x = []
    capture_te_all_x = []
    
    width_tr_all_x = []
    width_va_all_x = []
    width_te_all_x = []
    
    for ix in range(x_tr.shape[1]):
        #x_min = mu_pred_x_tr[:,ix] - 1.68 * sigma_pred_x_tr[:,ix]
        #x_max = mu_pred_x_tr[:,ix] + 1.68 * sigma_pred_x_tr[:,ix]
        
        if sigma_pred_x_tr.ndim == 3:
            x_min = mu_pred_x_tr[:,ix] - 1.68 * sigma_pred_x_tr[:,ix, ix]
            x_max = mu_pred_x_tr[:,ix] + 1.68 * sigma_pred_x_tr[:,ix, ix]
        else:
            print(mu_pred_x_tr.shape, sigma_pred_x_tr.shape)
            x_min = mu_pred_x_tr[:,ix] - 1.68 * sigma_pred_x_tr[:,ix]
            x_max = mu_pred_x_tr[:,ix] + 1.68 * sigma_pred_x_tr[:,ix]
        capture_tr_ix = np.mean((np.less(x_min, x_tr[:, ix]) * np.less(x_tr[:, ix], x_max)))
        capture_tr_all_x.append(capture_tr_ix)
        
        x_tr_range = (x_tr[:, ix].max() - x_tr[:, ix].min())
        width_tr = np.mean(np.abs(x_max - x_min)) / x_tr_range
        width_tr_all_x.append(width_tr)
        
        if sigma_pred_x_va.ndim == 3:
            x_min = mu_pred_x_va[:,ix] - 1.68 * sigma_pred_x_va[:,ix, ix]
            x_max = mu_pred_x_va[:,ix] + 1.68 * sigma_pred_x_va[:,ix, ix]
        else:
            x_min = mu_pred_x_va[:,ix] - 1.68 * sigma_pred_x_va[:,ix]
            x_max = mu_pred_x_va[:,ix] + 1.68 * sigma_pred_x_va[:,ix]
        capture_va_ix = np.mean((np.less(x_min, x_va[:, ix]) * np.less(x_va[:, ix], x_max)))
        capture_va_all_x.append(capture_va_ix)
        
        x_va_range = (x_va[:, ix].max() - x_va[:, ix].min())
        width_va = np.mean(np.abs(x_max - x_min)) / x_va_range
        width_va_all_x.append(width_va)
        
        if sigma_pred_x_te.ndim == 3:
            x_min = mu_pred_x_te[:,ix] - 1.68 * np.sqrt(sigma_pred_x_te[:,ix, ix])
            x_max = mu_pred_x_te[:,ix] + 1.68 * np.sqrt(sigma_pred_x_te[:,ix, ix])
        else:
            x_min = mu_pred_x_te[:,ix] - 1.68 * np.sqrt(sigma_pred_x_te[:,ix])
            x_max = mu_pred_x_te[:,ix] + 1.68 * np.sqrt(sigma_pred_x_te[:,ix])
            
        capture_te_ix = np.mean((np.less(x_min, x_te[:, ix]) * np.less(x_te[:, ix], x_max)))
        capture_te_all_x.append(capture_te_ix)
        
        x_te_range = (x_te[:, ix].max() - x_te[:, ix].min())
        width_te = np.mean(np.abs(x_max - x_min)) / x_te_range
        width_te_all_x.append(width_te)
        
    
    # percentage of captured points
    capture_tr = np.mean(capture_tr_all_x)
    capture_va = np.mean(capture_va_all_x)
    capture_te = np.mean(capture_te_all_x)
    
    # width of intervals
    width_tr = np.mean(width_tr_all_x)
    width_va = np.mean(width_va_all_x)
    width_te = np.mean(width_te_all_x)
    
    # print and store results
    print("{:<22} | {:<22} | {:<5} | {:.5f} {:.5f} {:.5f} |{:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} | {:<2} # {}".format(
            args.model_name, args.dataset, "exp_" + args.exp_num,
            pll_tr, pll_va, pll_te,
            mape_tr, mape_va, mape_te,
            capture_tr, capture_va, capture_te,
            width_tr, width_va, width_te,
            args.seed, 
            str(vars(args))))
    
    
    text_file = open("./results/{}".format(args.results_file), "a+")
    text_file.write("{:<22} | {:<22} | {:<5} |{:.5f} {:.5f} {:.5f} |{:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} |  {:.5f} {:.5f} {:.5f} | {:<2} # {}".format(
            args.model_name, args.dataset, "exp_" + args.exp_num,
            pll_tr, pll_va, pll_te,
            mape_tr, mape_va, mape_te,
            capture_tr, capture_va, capture_te,
            width_tr, width_va, width_te,
            args.seed, 
            str(vars(args))))
    text_file.write('\n') 
    text_file.close()
