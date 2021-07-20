from pypoptim.losses import RMSE

import logging
logger = logging.getLogger(__name__)


def calculate_loss(sol, config):

    loss = 0

    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue

        if config['loss'] == 'RMSE':
            x = sol['phenotype'][exp_cond_name]['I_out']  # TODO
            y = exp_cond['phenotype']['I_out']  # TODO
            loss += RMSE(x, y)

        else:
            raise NotImplementedError('I don\'t know your type of loss')

    logger.info(f'loss = {loss}')

    return loss
