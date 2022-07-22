import scipy.stats as ss
import scipy.special as ssp
import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from datetime import datetime
from fastprogress import fastprogress
fastprogress.printing = lambda: True
from pymc.step_methods.arraystep import ArrayStepShared, Competence, metrop_select
from pymc.step_methods.metropolis import MultivariateNormalProposal, NormalProposal, delta_logp
from pymc.blocking import DictToArrayBijection, RaveledVars
from typing import  Any, List, Tuple, Dict
from pymc.aesaraf import floatX
from pymc.step_methods.metropolis import tune

class StupidMetropolis(ArrayStepShared):
    name = "Stupid metropolis"
    print(f'Step name = {name}')
    default_blocked = False
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": bool,#np.float64,#bool,
            "tune": bool,
            "scaling": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        scaling=1.0,
        tune=True,
        tune_interval=100,
        model=None,
        mode=None,
        loglike = None,
        bounds = None, 
        beta_bounds = None,
        transform = True,
        **kwargs
    ):

        model = pm.modelcontext(model)
        initial_values = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
        vars = pm.inputvars(vars)
        
        initial_values_shape = [initial_values[v.name].shape for v in vars]
        if S is None:
            S = np.ones(int(sum(np.prod(ivs) for ivs in initial_values_shape)))
        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)

        self.scaling = np.atleast_1d(scaling).astype("d")
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (initial_values[v.name].size or 1) for v in vars]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # Metropolis will try to handle one batched dimension at a time This, however,
        # is not safe for discrete multivariate distributions (looking at you Multinomial),
        # due to high dependency among the support dimensions. For continuous multivariate
        # distributions we assume they are being transformed in a way that makes each
        # dimension semi-independent.
        is_scalar = len(initial_values_shape) == 1 and initial_values_shape[0] == ()
        self.elemwise_update = not (
            is_scalar
            or (
                self.any_discrete
                and max(getattr(model.values_to_rvs[var].owner.op, "ndim_supp", 1) for var in vars)
                > 0
            )
        )
        if self.elemwise_update:
            dims = int(sum(np.prod(ivs) for ivs in initial_values_shape))
        else:
            dims = 1
        self.enum_dims = np.arange(dims, dtype=int)
        self.accept_rate_iter = np.zeros(dims, dtype=float)
        self.accepted_iter = np.zeros(dims, dtype=bool)
        self.accepted_sum = np.zeros(dims, dtype=int)

        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(scaling=self.scaling, steps_until_tune=tune_interval)

        # TODO: This is not being used when compiling the logp function!
        self.mode = mode

        shared = pm.make_shared_replacements(initial_values, vars, model)
        self.delta_logp = delta_logp(initial_values, model.logp(), vars, shared)
        super().__init__(vars, shared)
        self.bounds = bounds
        self.beta_bounds = beta_bounds
        
        self.loglike = loglike
        self.transform = transform
        
    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values."""
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        self.accepted_sum[:] = 0
        return

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:
        point_map_info = q0.point_map_info
        q0 = q0.data
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(self.scaling, self.accepted_sum / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted_sum[:] = 0

        delta = self.proposal_dist() * self.scaling
        if not self.transform:
            while((q0 + delta <= self.beta_bounds[0]) or  (q0 + delta >= self.beta_bounds[1])): 
                delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q0 = q0.astype("int64")
                q = (q0 + delta).astype("int64")
            else:
                delta[self.discrete] = np.round(delta[self.discrete], 0)
                q = q0 + delta
        else:
            q = floatX(q0 + delta)
        
        if self.elemwise_update:
            q_temp = q0.copy()
            # Shuffle order of updates (probably we don't need to do this in every step)
            np.random.shuffle(self.enum_dims)
            for i in self.enum_dims:
                q_temp[i] = q[i]
                accept_rate_i = self.delta_logp(q_temp, q0)
                q_temp_, accepted_i = metrop_select(accept_rate_i, q_temp, q0)
                q_temp[i] = q_temp_[i]
                self.accept_rate_iter[i] = accept_rate_i
                self.accepted_iter[i] = accepted_i
                self.accepted_sum[i] += accepted_i
            q = q_temp
        else:
            accept_rate = self.delta_logp(q, q0)
            q, accepted = metrop_select(accept_rate, q, q0)
            self.accept_rate_iter = accept_rate
            self.accepted_iter = accepted
            self.accepted_sum += accepted

        self.steps_until_tune -= 1
        stats = {
            "tune": self.tune,
            "scaling": np.mean(self.scaling),
            "accept": np.mean(np.exp(self.accept_rate_iter)),
            "accepted": np.mean(self.accepted_iter),
        }
        
        return RaveledVars(q, point_map_info), [stats]
    
    def step(self, point):

        for name, shared_var in self.shared.items():
            shared_var.set_value(point[name])

        q = DictToArrayBijection.map({v.name: point[v.name] for v in self.vars})

        step_res = self.astep(q)

        if self.generates_stats:
            apoint, stats = step_res
        else:
            apoint = step_res

        if not isinstance(apoint, RaveledVars):
            # We assume that the mapping has stayed the same
            apoint = RaveledVars(apoint, q.point_map_info)

        new_point = DictToArrayBijection.rmap(apoint, start_point=point)
        beta_key, param_key = list(new_point.keys())
        beta_new, params_new = new_point[beta_key], new_point[param_key]

        if self.transform:
            beta_new = (self.beta_bounds[1]*np.exp(beta_new) + self.beta_bounds[0])/(1 + np.exp(beta_new))
            params_new = (self.bounds[1]*np.exp(params_new) + self.bounds[0])/(1 + np.exp(params_new))
        self.loglike.update_cov_matrix(beta_accepted=beta_new, params_accepted=params_new)
        if self.generates_stats:
            return new_point, stats
        return new_point

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE

class StupidDEMetropolisZ(ArrayStepShared):
    name = "StupidDEMetropolisZ"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": bool,
            "tune": bool,
            "scaling": np.float64,
            "lambda": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,
        tune="lambda",
        tune_interval=100,
        tune_drop_fraction: float = 0.9,
        model=None,
        mode=None,
        bounds=None,
        initial_values_size=None,
        transform=True,
        **kwargs
    ):
        model = pm.modelcontext(model)
        initial_values = model.initial_point()

        if initial_values_size is None:
            initial_values_size = sum(initial_values[n.name].size 
            for n in model.value_vars)#2

        if vars is None:
            vars = model.cont_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(initial_values_size)
        
        self.S = S
        
        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(self.S)
        else:
            self.proposal_dist = UniformProposal(self.S)
        
        self.scaling = np.atleast_1d(scaling).astype("d")
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * initial_values_size)
        self.lamb = float(lamb)
        if tune not in {None, "scaling", "lambda", "S"}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda, S}')
        self.tune = True
        self.tune_target = tune
        self.tune_interval = tune_interval
        self.tune_drop_fraction = tune_drop_fraction
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # cache local history for the Z-proposals
        self._history = []
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling,
            lamb=self.lamb,
            steps_until_tune=tune_interval,
            accepted=self.accepted,
        )

        self.mode = mode
        self.bounds = bounds
        self.initial_values_size = initial_values_size
        self.transform = transform
        
        shared = pm.make_shared_replacements(initial_values, vars, model)
        self.delta_logp = delta_logp(initial_values, model.logp(), vars, shared)
        super().__init__(vars, shared)


    def reset_tuning(self):
        """Resets the tuned sampler parameters and history to their initial values."""
        # history can't be reset via the _untuned_settings dict because it's a list
        self._history = []
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:

        point_map_info = q0.point_map_info
        q0 = q0.data
        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune: #and self.tune:
            if self.tune_target == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
               
            elif self.tune_target == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            elif self.tune_target == "S":

                eps = 0.00001
                self.S = 2.38**2/self.initial_values_size*(np.cov(self._history,rowvar=False)+eps*np.identity(self.initial_values_size)) 
                self.proposal_dist = pm.MultivariateNormalProposal(self.S)

                
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling
        if not self.transform:
            while(np.any(q0 + epsilon <= self.bounds[0]) or np.any(q0 + epsilon >= self.bounds[1])): 
                epsilon = self.proposal_dist() * self.scaling

        it = len(self._history)
        # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries
        if it > 1:
            # differential evolution proposal
            # select two other chains
            iz1 = np.random.randint(it)
            iz2 = np.random.randint(it)
            while iz2 == iz1:
                iz2 = np.random.randint(it)

            z1 = self._history[iz1]
            z2 = self._history[iz2]
            # propose a jump
            q = floatX(q0 + self.lamb * (z1 - z2) + epsilon)
        else:
            # propose just with noise in the first 2 iterations
            q = floatX(q0 + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        if not accepted: #Delayed Rejection
            epsilon_2 = self.proposal_dist() * self.scaling/5
            if not self.transform:
                while(np.any(q0 + epsilon_2 <= self.bounds[0]) or np.any(q0 + epsilon_2 >= self.bounds[1])): 
                    epsilon_2 = self.proposal_dist() * self.scaling/5
            
            q2 = floatX(q0 + epsilon_2)
            dellog_q_q2 = self.delta_logp(q, q2) #gonna use it below. TODO: still one excessive evaluation of loglike.

            if dellog_q_q2 > 0: 
                accept2=-np.inf
            else:
                alpha = 1 - np.exp(dellog_q_q2)
                alpha_ratio = np.log(alpha)-np.log(1-np.exp(accept))
                scaled_S_inv = np.linalg.inv(self.scaling**2*self.S) #accounted for scaling
                prop_ratio = (-np.dot(q2-q,np.dot(scaled_S_inv,q2-q))+np.dot(q0-q,np.dot(scaled_S_inv,q0-q)))/2
                accept2 = accept-dellog_q_q2+prop_ratio+alpha_ratio 
                
            q_new, accepted = metrop_select(accept2, q2, q0)
            
        self.accepted += accepted
        self._history.append(q_new)

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        q_new = RaveledVars(q_new, point_map_info)

        return q_new, [stats]


    def stop_tuning(self):
        """At the end of the tuning phase, this method removes the first x% of the history
        so future proposals are not informed by unconverged tuning iterations.
        """
        it = len(self._history)
        n_drop = int(self.tune_drop_fraction * it)
        self._history = self._history[n_drop:]
        return super().stop_tuning()


    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE

    
def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    return scale * np.where(
        acc_rate < 0.001,
        # reduce by 90 percent
        0.1,
        np.where(
            acc_rate < 0.05,
            # reduce by 50 percent
            0.5,
            np.where(
                acc_rate < 0.2,
                # reduce by ten percent
                0.9,
                np.where(
                    acc_rate > 0.95,
                    # increase by factor of ten
                    10.0,
                    np.where(
                        acc_rate > 0.75,
                        # increase by double
                        2.0,
                        np.where(
                            acc_rate > 0.5,
                            # increase by ten percent
                            1.1,
                            # Do not change
                            1.0,
                        ),
                    ),
                ),
            ),
        ),
    )

class AwfulLogLike(at.Op):
    
    itypes = [at.dscalar, at.dvector]
    otypes = [at.dscalar] 

    def __init__(self, 
                 ina_dict=None, 
                 loglike=None, 
                 n=None, 
                 p=None, 
                 delta=None, 
                 S=None,
                 phi=None,
                 cov_mat=None,
                 beta_fixed=None,
                ):

        print('Updating beta, phi, cov_mat from inverse wishart distribution')
        
        self.find_S = ina_dict['find_S']
        self.data = ina_dict['data']
        self.m_index = ina_dict['m_index']
        self.Ina = ina_dict['Ina']
        self.const = ina_dict['const']
        self.config = ina_dict['config']
        
        self.likelihood = loglike
        self.n = n
        self.p = p
        self.delta = delta

        self.S = S
        self.phi = phi
        self.cov_mat = cov_mat
        self.beta = beta_fixed
        
        
    def perform(self, node, inputs, outputs,):
        (beta, params) = inputs 
      
        self.S = self.find_S(params, 
                             data = self.data,
                             m_index=self.m_index, 
                             Ina = self.Ina,
                             const=self.const,
                             config=self.config,)
                                                
        if type(self.S) != np.ndarray:
            logl = np.array(self.S)
        else:
            logl = self.likelihood(beta,
                                   S=self.S,
                                   cov=self.cov_mat,
                                   phi=self.phi,
                                   p=self.p,
                                   n=self.n,
                                   delta=self.delta)

        outputs[0][0] = np.array(logl)

    def update_cov_matrix(self, beta_accepted=None, params_accepted=None):

        self.S = self.find_S(params_accepted, 
                             self.data,
                             m_index=self.m_index, 
                             Ina=self.Ina,
                             const=self.const,
                             config=self.config,
                             )

        self.cov_mat = ss.invwishart.rvs(df=beta_accepted + self.n,
                                        scale=self.S + self.phi
                                        )

        invcov_diag = np.diag(np.linalg.inv(self.cov_mat))

        self.phi = np.diag([ss.gamma.rvs(beta_accepted/2,
                                        scale = 2/invcov_diag[k])
                                        for k in range(self.p)]
                                        )