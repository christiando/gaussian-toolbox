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
import statsmodels.api as sm
from darts.models import TCNModel
from darts.models import GaussianProcessFilter
from darts.utils.likelihood_models import GaussianLikelihoodModel
from darts.timeseries import TimeSeries
from exp_utils import *

import newt
import objax
from ssm import HMM

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

class HMM_class:
    
    def __init__(self, x_tr, K, obs_model='gaussian'):
        self.x_tr = x_tr
        self.D = x_tr.shape[1]
        self.K = K
        self.obs_model = obs_model
        self.model = self._train()
        
    def _train(self):
        model = HMM(self.K, self.D, observations=self.obs_model)
        model.fit(self.x_tr, method="em")
        return model

    def compute_predictive_log_likelihood(self, x_te):
        return self.model.log_likelihood(x_te)
    
    def compute_predictive_density(self, x_te):
        mask = np.logical_not(np.isnan(x_te))
        x_te_not_nan = np.zeros(x_te.shape)
        x_te_not_nan[mask] = x_te[mask]
        states = self.model.filter(x_te_not_nan, mask=mask)
        if self.obs_model == 'gaussian':
            mean_te = np.dot(states, self.model.observations.mus)
        elif self.obs_model == 'ar':
            mean_te = np.sum(states[:,:,None] * (np.sum(self.model.observations.As[None] * x_te[:,None, None], axis=3) + self.model.observations.bs), axis=1)
        std_te = np.dot(states, np.sqrt(self.model.observations.Sigmas.diagonal(axis1=1, axis2=2)))
        print(mean_te.shape, std_te.shape)
        return PredictiveDensity(mean_te, std_te)

def train_HMM(x_tr, **kwargs):
    return HMM_class(x_tr, args.num_states, args.obs_model)
    
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
        var_f2 = 3.  # GP variance
        len_f2 = 5.  # GP lengthscale
        
        if args.newt_kernel == 'Matern12':
            kern1 = newt.kernels.Matern12(variance=var_f1, lengthscale=len_f1)
            kern2 = newt.kernels.Matern12(variance=var_f2, lengthscale=len_f2)
        elif args.newt_kernel == 'Matern32':
            kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
            kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)

        kern = newt.kernels.Independent([kern1, kern2])
        lik = newt.likelihoods.HeteroscedasticNoise(link=args.newt_link)
        #model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        model = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        lr_adam = 0.01
        lr_newton = 0.05
        iters = 200
        opt_hypers = objax.optimizer.Adam(model.vars())
        energy = objax.GradValues(model.energy, model.vars())
        e2 = np.inf
        converged = False

        @objax.Function.with_vars(model.vars() + opt_hypers.vars())
        def train_op():
            model.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

        train_op = objax.Jit(train_op)
        i= 0
        while not converged:
            loss = train_op()
            e1 = loss[0]
            converged = (np.abs(e2 - e1) / np.amax(np.abs([e1, e2]))) < 1e-4
            e2 = e1
            i += 1
        return model
    
    def compute_predictive_log_likelihood(self, x_te):
        predictions = self.compute_predictive_density(x_te)
        llk = - .5 * np.sum(((x_te - predictions.mu) / predictions.Sigma) ** 2 + np.log(2 * np.pi * predictions.Sigma ** 2))
        return llk
        #return model_te.compute_log_lik()
    
    def compute_predictive_density(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        x_te_nan_idx = np.where([np.any(np.isnan(x_te), axis=1)])[1]
        x_te_not_nan_idx = np.where([np.logical_not(np.any(np.isnan(x_te), axis=1))])[1]
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

        if args.newt_kernel == 'Matern12':
            kern1 = newt.kernels.Matern12(variance=var_f1, lengthscale=len_f1)
            kern2 = newt.kernels.Matern12(variance=var_f2, lengthscale=len_f2)
        elif args.newt_kernel == 'Matern32':
            kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
            kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)

        kern = newt.kernels.Independent([kern1, kern2])
        lik = newt.likelihoods.HeteroscedasticNoise(link=args.newt_link)
        #model_te = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        model_te = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        lr_newton = 0.05
        e2 = np.inf
        converged = False
        #for i in range(100):
        i = 0
        opt_hypers = objax.optimizer.Adam(model_te.vars())
        energy = objax.GradValues(model_te.energy, model_te.vars())
        
        @objax.Function.with_vars(model_te.vars() + opt_hypers.vars())
        def train_op():
            model_te.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            #opt_hypers(lr_adam, dE)
            return E[0]

        train_op = objax.Jit(train_op)
        
        while not converged:
            e1 = train_op()
            #e1 = model_te.energy()
            converged = (np.abs(e2 - e1) / np.amax(np.abs([e1, e2]))) < 1e-4
            e2 = e1
            i += 1
        """
        lr_adam = 0.01
        
        
        opt_hypers = objax.optimizer.Adam(model_te.vars())
        energy = objax.GradValues(model_te.energy, model_te.vars())

        @objax.Function.with_vars(model_te.vars() + opt_hypers.vars())
        def train_op():
            model_te.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E
        
        for i in range(1, 10 + 1):
            loss = train_op()
            print(loss)
        """
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

        if args.newt_kernel == 'Matern12':
            kern1 = newt.kernels.Matern12(variance=var_f1, lengthscale=len_f1)
        elif args.newt_kernel == 'Matern32':
            kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        #kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, ])
        #lik = newt.likelihoods.HeteroscedasticNoise()
        lik = newt.likelihoods.Gaussian()
        #model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
        model = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)

        lr_adam = 0.01
        lr_newton = 0.05
        e2 = np.inf
        converged = False
        #for i in range(100):
        i = 0
        opt_hypers = objax.optimizer.Adam(model.vars())
        energy = objax.GradValues(model.energy, model.vars())


        @objax.Function.with_vars(model.vars() + opt_hypers.vars())
        def train_op():
            model.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

        train_op = objax.Jit(train_op)

        while i < 200:
            loss = train_op()
            e1 = loss[0]
            converged = (np.abs(e2 - e1) / np.amax(np.abs([e1, e2]))) < 1e-4
            e2 = e1
            i += 1
        print(i)
        return model
    
    def compute_predictive_log_likelihood(self, x_te):
        #t_te = np.array([np.arange(x_te.shape[0])]).T
        #model_te = self._train_test_model(x_te, t_te)
        predictions = self.compute_predictive_density(x_te)
        llk = - .5 * np.sum(((x_te - predictions.mu) / predictions.Sigma) ** 2 + np.log(2 * np.pi * predictions.Sigma ** 2))
        return llk#model_te.compute_log_lik()
    
    def compute_predictive_density(self, x_te):
        t_te = np.array([np.arange(x_te.shape[0])]).T
        x_te_nan_idx = np.where([np.any(np.isnan(x_te), axis=1)])[1]
        x_te_not_nan_idx = np.where([np.logical_not(np.any(np.isnan(x_te), axis=1))])[1]
        model_te = self._train_test_model(x_te[x_te_not_nan_idx], t_te[x_te_not_nan_idx])
        mean_te, std_te = model_te.predict_y(t_te)
        return PredictiveDensity(mean_te, std_te)
    
    def _train_test_model(self, x_te, t_te):
        var_f1 = self.model.kernel.kernel0.variance  # GP variance
        len_f1 = self.model.kernel.kernel0.lengthscale  # GP lengthscale
        #var_f2 = self.model.kernel.kernel1.variance  # GP variance
        #len_f2 = self.model.kernel.kernel1.lengthscale  # GP lengthscale

        if args.newt_kernel == 'Matern12':
            kern1 = newt.kernels.Matern12(variance=var_f1, lengthscale=len_f1)
        elif args.newt_kernel == 'Matern32':
            kern1 = newt.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
        #kern2 = newt.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
        kern = newt.kernels.Independent([kern1, ])
        #lik = newt.likelihoods.HeteroscedasticNoise()
        lik = newt.likelihoods.Gaussian()
        lik.transformed_variance =  self.model.likelihood.transformed_variance
        t_te = np.array([np.arange(x_te.shape[0])]).T
        #model_te = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        model_te = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=t_te, Y=x_te)
        e2 = np.inf
        converged = False
        #for i in range(100):
        i = 0
        opt_hypers = objax.optimizer.Adam(model_te.vars())
        energy = objax.GradValues(model_te.energy, model_te.vars())
        lr_adam = 0.01
        lr_newton = 0.05
        
        @objax.Function.with_vars(model_te.vars() + opt_hypers.vars())
        def train_op():
            model_te.inference(lr=lr_newton, **self.inf_args)  # perform inference and update variational params
            dE, E = energy(**self.inf_args)  # compute energy and its gradients w.r.t. hypers
            #opt_hypers(lr_adam, dE)
            return E[0]
        
        train_op = objax.Jit(train_op)
        
        while not converged:
            e1 = train_op()
            #e1 = model_te.energy()
            converged = (np.abs(e2 - e1) / np.amax(np.abs([e1, e2]))) < 1e-4
            e2 = e1
            i += 1
            
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





def compute_mape_old(s_true, s_pred):
    
    if isinstance(s_true, darts.timeseries.TimeSeries):
        res = darts.metrics.mape(s_pred, s_true) 
    else:
        res = mape(s_true, s_pred)
    
    return res

def compute_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100

class DynamicFactor_ext():
    def __init__(self, x_tr):
        
        x_tr_df = pd.DataFrame(x_tr)
        self.dyn_fact_model = sm.tsa.DynamicFactor(x_tr_df, k_factors=1, factor_order=1)
        self.trained_model_result = self.dyn_fact_model.fit(full_output=False)
        
    def compute_predictive_density(self, x_te):
        dyn_fact_model_te = self.dyn_fact_model.clone(x_te)
        smoothed_result = dyn_fact_model_te.smooth(params=self.trained_model_result.params, return_ssm=True)
        #predictions = res.forecasts.T
        smoothed_result.mu = smoothed_result.forecasts.T
        smoothed_result.Sigma = smoothed_result.obs_cov.reshape(1, x_te.shape[1], x_te.shape[1])
        return smoothed_result
    
    def compute_predictive_log_likelihood(self, x_te):
        dyn_fact_model_te = self.dyn_fact_model.clone(x_te)
        smoothed_result = dyn_fact_model_te.smooth(params=self.trained_model_result.params, return_ssm=True)
        
        return np.asarray(dyn_fact_model_te.loglikeobs(params=self.trained_model_result.params))[:-1].sum()

    
        
def train_dyn_factor(x_tr):
    
    dyn_fact_model = DynamicFactor_ext(x_tr)
    
    return dyn_fact_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="energy")
    parser.add_argument('--model_name', type=str, default="dyn_factor")
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
<<<<<<< timeseries/experiments_scripts/run_experiment.py
    parser.add_argument('--exp_num', type=str, default="2")
=======
    parser.add_argument('--exp_num', type=str, default="1")
    parser.add_argument('--newt_kernel', type=str, default="Matern12")
    parser.add_argument('--newt_link', type=str, default="softplus")
    parser.add_argument('--num_states', type=int, default=1)
    parser.add_argument('--obs_model', type=str, default='gaussian')
>>>>>>> timeseries/experiments_scripts/run_experiment.py
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
    if args.model_name == 'dyn_factor':
        model =  'dyn_factor'
    if args.model_name == 'newt_gauss':
        model = 'newt_gauss'
    if args.model_name == 'newt_hsk':
        model = 'newt_hsk'
    if args.model_name == 'hmm':
        model = 'HMM'
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

    # compute metrics
    mape_tr = compute_mape(x_tr, mu_pred_x_tr)#compute_mape(x_tr, mu_pred_x_tr)
    mape_va = compute_mape(x_va, mu_pred_x_va)
    mape_te = compute_mape(x_te, mu_pred_x_te)

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
            capture_tr_ix = np.nanmean((np.less(x_min, x_tr[:, ix]) * np.less(x_tr[:, ix], x_max)))
        else:
            print(mu_pred_x_tr.shape, sigma_pred_x_tr.shape)
            x_min = mu_pred_x_tr[:,ix] - 1.68 * sigma_pred_x_tr[:,ix]
            x_max = mu_pred_x_tr[:,ix] + 1.68 * sigma_pred_x_tr[:,ix]
        capture_tr_ix = np.nanmean((np.less(x_min, x_tr[:, ix]) * np.less(x_tr[:, ix], x_max)))

        capture_tr_all_x.append(capture_tr_ix)
        
        x_tr_range = (x_tr[:, ix].max() - x_tr[:, ix].min())
        width_tr = np.nanmean(np.abs(x_max - x_min)) / x_tr_range
        width_tr_all_x.append(width_tr)
        
        if sigma_pred_x_va.ndim == 3:
            x_min = mu_pred_x_va[:,ix] - 1.68 * sigma_pred_x_va[:,ix, ix]
            x_max = mu_pred_x_va[:,ix] + 1.68 * sigma_pred_x_va[:,ix, ix]
        else:
            x_min = mu_pred_x_va[:,ix] - 1.68 * sigma_pred_x_va[:,ix]
            x_max = mu_pred_x_va[:,ix] + 1.68 * sigma_pred_x_va[:,ix]
        capture_va_ix = np.nanmean((np.less(x_min, x_va[:, ix]) * np.less(x_va[:, ix], x_max)))
        capture_va_all_x.append(capture_va_ix)
        
        x_va_range = (x_va[:, ix].max() - x_va[:, ix].min())
        width_va = np.nanmean(np.abs(x_max - x_min)) / x_va_range
        width_va_all_x.append(width_va)
        
        if sigma_pred_x_te.ndim == 3:
            x_min = mu_pred_x_te[:,ix] - 1.68 * np.sqrt(sigma_pred_x_te[:,ix, ix])
            x_max = mu_pred_x_te[:,ix] + 1.68 * np.sqrt(sigma_pred_x_te[:,ix, ix])
        else:
            x_min = mu_pred_x_te[:,ix] - 1.68 * np.sqrt(sigma_pred_x_te[:,ix])
            x_max = mu_pred_x_te[:,ix] + 1.68 * np.sqrt(sigma_pred_x_te[:,ix])
            
        capture_te_ix = np.nanmean((np.less(x_min, x_te[:, ix]) * np.less(x_te[:, ix], x_max)))
        capture_te_all_x.append(capture_te_ix)
        
        x_te_range = (x_te[:, ix].max() - x_te[:, ix].min())
        width_te = np.nanmean(np.abs(x_max - x_min)) / x_te_range
        width_te_all_x.append(width_te)
        
    
    # percentage of captured points
    capture_tr = np.nanmean(capture_tr_all_x)
    capture_va = np.nanmean(capture_va_all_x)
    capture_te = np.nanmean(capture_te_all_x)
    
    # width of intervals
    width_tr = np.nanmean(width_tr_all_x)
    width_va = np.nanmean(width_va_all_x)
    width_te = np.nanmean(width_te_all_x)
    

    
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
