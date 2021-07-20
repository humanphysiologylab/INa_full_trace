import numpy as np
import pandas as pd

from pypoptim.algorythm import Solution

from gene_utils import update_S_C_from_genes

from loss_utils import calculate_loss


class SolModel(Solution):

    def __init__(self, x, **kwargs_data):
        super().__init__(x, **kwargs_data)
        for attr in 'model', 'config':
            if not hasattr(self, attr):
                raise AttributeError(attr, "make this guy static")

        self._status = None
        self.__status_valid = 2

    @property
    def status(self):
        return self._status

    def update(self):

        self['phenotype'] = {}

        legend = self.config['runtime']['legend']

        genes = pd.Series(self.x, index=self.config['runtime']['m_index'])

        for exp_cond_name in self.config['experimental_conditions']:

            if exp_cond_name == 'common':
                continue

            C = legend['constants'].copy()
            # S = self.config['state']  # ??? TODO, do you need S to be put in self.model.run()?
            S = legend['states'].copy() # DONE
            A = legend['algebraic'].copy()

            update_S_C_from_genes(S, C, genes, exp_cond_name, self.config)

            df_protocol = self.config['runtime']['protocol']  # DONE
            df_initial_state_protocol = self.config['runtime']['initial_state_protocol']  # DONE

            pred = self.model.run(A,
                                  S,  # <- TODO: do you need this?
                                  C,
                                  df_protocol,  # DONE
                                  df_initial_state_protocol,  # DONE
                                  **self.config)
            self._status = self.model.status
            if self._status != self.__status_valid:
                self._x = genes.values
                self._y = np.nan
                return
            self['phenotype'][exp_cond_name] = pred.copy()

        self._x = genes.values
        self._y = calculate_loss(self, self.config)

    def is_valid(self):
        if not self.is_updated():
            return False
        else:
            flag_valid = self._status == self.__status_valid and np.isfinite(self._y)
            if 'phenotype' not in self:  # solution was gathered via MPI
                return flag_valid
            else:
               return flag_valid and all(not np.any(np.isnan(p)) for p in self['phenotype'].values())
