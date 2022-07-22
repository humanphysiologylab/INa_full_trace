from distutils.dep_util import newer_group
from re import M
import numpy as np
import pandas as pd
import sys
sys.path.append('../../pypoptim/mpi_scripts/')
from gene_utils import update_C_from_genes
import scipy.special as ssp


def beta_loglike(beta,
                cov=None, 
                phi=None, 
                p=None, 
                delta=None,
                ):

    sign, logdet = np.linalg.slogdet(cov)
    C = (-logdet + p * (np.log(phi) - np.log(2)))/2
    loss = C*beta-delta*np.log(beta)-ssp.multigammaln(beta/2,int(p))
    return loss


def phi_loglike(beta,
                S=None,
                cov=None, 
                phi=None, 
                p=None, 
                n=None, 
                delta=None,
                ):

    sign_cov, logdet_cov=np.linalg.slogdet(cov)
    matrix_part = (-np.trace(np.dot(np.linalg.inv(cov), np.diag([phi for k in range(p)]) + S)) - (beta + n + p + 1) * logdet_cov + (p * beta) * np.log(phi)) / 2
    const_part = -(np.log(2 * np.pi) * n + np.log(2) * beta) * p/2 - delta * np.log(beta) - np.log(phi)
    loss = matrix_part + const_part - ssp.multigammaln(beta/2, p)
    return loss
    

def big_loglike(beta,
            S=None,
            cov=None,
            phi=None,
            p=None,
            n=None,
            delta=None,
            ):

    sign_cov, logdet_cov=np.linalg.slogdet(cov)
    matrix_part = (-np.trace(np.dot(np.linalg.inv(cov), (phi + S))) 
                    - (beta + n + p + 1) * logdet_cov) / 2  + (beta/2 - 1) * np.sum(np.log(np.diag(phi)))
    const_part = -(np.log(2 * np.pi) * n + np.log(2) * beta) * p/2 - delta * np.log(beta) 
    loss = matrix_part + const_part - ssp.multigammaln(beta/2, p)
    return loss

def func_model(params, 
               m_index,
               Ina=None,
               const=None,
               config=None,
               ):
    C = const.copy()
    A = config['runtime']['legend']['algebraic'].copy()
    S = config['runtime']['legend']['states'].copy()
    
    df_protocol = config['experimental_conditions']['trace']['protocol'] 
    df_initial_state_protocol = config['runtime']['initial_state_protocol']

    genes = pd.Series(params, index=m_index)
    update_C_from_genes(C, genes, 'trace', config)
    ina = np.array(Ina.run(A,S,C,df_protocol, df_initial_state_protocol, 20).I_out)
    return ina


def return_data_cut(Params,
                    data=None, 
                    m_index=None, 
                    Ina=None, 
                    const=None, 
                    config=None,
                    len_step=5000,
                    null_ind=np.array([0, 7, 107, 207, 407, 500]),
                    pipette_ind=[7, 107, 207, 407],
                    downsampling=None,
                    mask_mult=None,
                    mask_cut=None,
                    ):
    params=Params.copy()
    if mask_mult is not None:
        params[mask_mult] = 10**params[mask_mult]
    if mask_cut is None:
        mask_cut = np.ones(len_step).astype('bool')

    ina = func_model(params, m_index, Ina=Ina, const=const, config=config)  
    if np.any(np.isnan(ina)):
        return np.float64(-1e50)

    delta_ina = data - ina    
    data_cut_size = np.sum(mask_cut)
    data_cut = np.zeros([19, data_cut_size])
    
    for k in range(19):
        data_cut[k] = delta_ina[(k+1)*len_step:(k+2)*len_step:downsampling][mask_cut]
    return data_cut

def find_S(params,
            data, 
            m_index=None, 
            Ina=None, 
            const=None, 
            config=None,
            len_step=5000,
            downsampling=10,
            mask_mult=None,
            mask_cut=None,
            ):
    data_cut = return_data_cut(params,
                                data=data, 
                                m_index=m_index, 
                                Ina=Ina, 
                                const=const, 
                                config=config,
                                len_step=len_step,
                                downsampling=downsampling,
                                mask_mult=mask_mult,
                                mask_cut=mask_cut,

                                )
    if type(data_cut) != np.ndarray:
        return data_cut
    S0 = np.dot(data_cut.T, data_cut)
    # S_changed = np.diag(np.diag(S0)) + np.diag(np.diag(S0, k=1), k=1)+ np.diag(np.diag(S0, k=-1), k=-1) #+ np.diag(np.diag(S0, k=2), k=2) + np.diag(np.diag(S0, k=-2), k=-2)
    # null_vec = np.zeros(data_cut_size)
    # S_changed = np.zeros([data_cut_size, data_cut_size])
    # for i in range(len(null_ind)-1):
    #     null_vec_now = null_vec.copy()
    #     null_vec_now[null_ind[i]:null_ind[i+1]] = 1
    #     S_changed += np.dot(np.diag(null_vec_now), np.dot(S0, np.diag(null_vec_now)))
            
    return S0