from pypoptim.losses import RMSE
import numpy as np
import matplotlib.pyplot as plt

def show_fig(datas, 
             name = None, 
             *,
             dt = 5e-5,
             nrows = 4, 
             ncols = 5,
             color_reference="0.2",
             ls_reference="-",
             lss = None,
             lw_reference = 3,
             lws = None,
             colors = None,
             xlim = None, 
             ylim = None,
             n_sections = 20,
             all_len = 100000,
             start = -80, 
             step = 5,
             i_global = 0,
             i_global_step =1,
             ylabel = 'I, pA',
             ):
    
    if colors is None:
        colors = [color_reference] + [f'C{i}' for i in range(len(datas) - 1)]
    else:
        colors = [color_reference] + colors
    
    if lss is None:
        lss = [ls_reference] + ['-' for i in range(len(datas) -1)]
    else:
        lss = [ls_reference] + lss
        
    if lws is None:
        lws = [lw_reference] + [2 for i in range(len(datas) -1)]
    else:
        lws = [lw_reference] + lws
    
    
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols,
                             figsize=plt.figaspect(nrows / ncols) * 1.5,
                             sharex=True)
    fig.suptitle(name)
    
    len_trace = int(all_len/n_sections)
    split_indices = np.linspace(0, all_len, n_sections + 1).astype(int)
    
    t = np.arange(len_trace) * dt
    
    if xlim is None:
        xlim = (0.,len_trace * dt )
 

        
    for i_row in range(nrows):
        for i_col in range(ncols):
            ax = axes[i_row, i_col]
            plt.sca(ax)
            i_start, i_end = split_indices[i_global], split_indices[i_global + 1]
            slice_current = slice(i_start, i_end)
            
            for k, data in enumerate(datas):
                data_show = data[slice_current]
                index_reference = 0
                data_reference = datas[0][slice_current]
                
                loss = RMSE(data_show, data_reference)
                plt.text(1, 1 - k * 0.05, f"{loss:.2f}", fontsize="x-small",
                         transform=ax.transAxes, ha="right", color=colors[k])
                
                plt.plot(t, data_show, color=colors[k], ls = lss[k], lw = lws[k])
                if ylim is None:
                    plt.ylim(min(data_show)-100, max(data_show)+50)
                else:
                    plt.ylim(ylim)
            
            step_value = start + step * i_global
            title = f'step {step_value} mV'
            plt.title(title)
            plt.xlim(xlim)
            i_global += i_global_step
            if i_col == 0:
                plt.ylabel(ylabel)
            if i_row == nrows-1:
                plt.xlabel('t, s')
                      
