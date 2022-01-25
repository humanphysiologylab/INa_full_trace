from pypoptim.losses import RMSE
import numpy as np
import logging
logger = logging.getLogger(__name__)


def calculate_loss(sol, config):

    loss = 0
    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue
        if config['loss'] == 'RMSE':
            x = sol['phenotype'][exp_cond_name]['I_out']
            y = exp_cond['phenotype']['I_out']
            sample_weight = exp_cond.get('sample_weight', None)
            loss += RMSE(x, y, sample_weight=sample_weight)
        elif config['loss'] == 'RMSE_GRAD':
            x = sol['phenotype'][exp_cond_name]['I_out']
            x_grad = sol['phenotype'][exp_cond_name]['grad']
            y = exp_cond['phenotype']['I_out']
            y_grad = np.gradient(y)
            sample_weight = exp_cond.get('sample_weight', None)
            sample_weight_grad = exp_cond.get('sample_weight_grad', None)
            loss += RMSE(x, y, sample_weight=sample_weight)/10
            loss += RMSE(x_grad, y_grad, sample_weight=sample_weight_grad)
        else:
            raise NotImplementedError('I don\'t know your type of loss')

    logger.info(f'loss = {loss}')

    return loss


def calculate_loss_another(sol, config):

    loss_trace = 0
    loss_grad = 0
    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue
        if config['loss'] == 'RMSE':
            x = sol['phenotype'][exp_cond_name]['I_out']
            y = exp_cond['phenotype']['I_out']
            sample_weight = exp_cond.get('sample_weight', None)
            loss_trace += RMSE(x, y, sample_weight=sample_weight)
        elif config['loss'] == 'RMSE_GRAD':
            x = sol['phenotype'][exp_cond_name]['I_out']
            x_grad = sol['phenotype'][exp_cond_name]['grad']
            y = exp_cond['phenotype']['I_out']
            y_grad = np.gradient(y)
            sample_weight = exp_cond.get('sample_weight', None)
            sample_weight_grad = exp_cond.get('sample_weight_grad', None)
            loss_trace += RMSE(x, y, sample_weight=sample_weight)/10
            loss_grad += RMSE(x_grad, y_grad, sample_weight=sample_weight_grad)
        else:
            raise NotImplementedError('I don\'t know your type of loss')

    logger.info(f'loss = {loss_trace}')

    return loss_trace, loss_grad
