import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from scipy.stats import spearmanr

def plot_train_dev_metric(epochs, train_metric, eval_metric, base_path, metric_name, dataset_name):
    plt.plot(epochs, train_metric, '#3fc1fd', label='Train')
    plt.plot(epochs, eval_metric, '#d09fff', label='Validation')
    # plt.plot([330, 330], [0.9773016059994697-0.1, 1.0378530149936677+0.1], '#fd8989', label='Take the model parameters of the epoch')
    plt.title('Train and Validation {} on {}'.format(metric_name, dataset_name))
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(base_path, dataset_name + '_' + metric_name +'.jpg'))
    plt.cla()


def cal_mean_std(transfer_results):
    mean_transfer_results = np.zeros(transfer_results.shape[:2]) # np.zeros_like(transfer_results)
    std_transfer_results = np.zeros(transfer_results.shape[:2]) # np.zeros_like(transfer_results)
    for tgt_idx in range(transfer_results.shape[1]): # square shape
        for src_idx in range(transfer_results.shape[0]):
            mean_transfer_results[src_idx, tgt_idx] = np.mean(transfer_results[src_idx, tgt_idx])
            std_transfer_results[src_idx, tgt_idx] = np.std(transfer_results[src_idx, tgt_idx])
    return mean_transfer_results.astype(np.float32), std_transfer_results.astype(np.float32)


def calcul_transfer_stats(orig_ft_results, transfer_results, tr_rows, tr_cols, ft_inter_dists):
    mean_transfer_results, std_transfer_results = cal_mean_std(transfer_results)

    mean_transfer_results = pd.DataFrame(mean_transfer_results, columns=tr_cols)
    std_transfer_results = pd.DataFrame(std_transfer_results, columns=tr_cols)

    # Performance迁移结果比值
    transfer_ratio = (mean_transfer_results - orig_ft_results) / orig_ft_results * 100 # -> %
    transfer_ratio['row_name'] = tr_rows
    # Performance迁移结果error margin（std ratio）
    transfer_error = std_transfer_results / orig_ft_results * 100
    transfer_error['row_name'] = tr_rows
    
    # inter-cosine距离
    ft_inter_dists['row_name'] = tr_rows

    # 构建迁移stats
    abbr_dict = {'PDBBind': 'PDBBind', 'Kinase': 'Kinase', 'DUDE': "DUDE"}
    transfer_stats = pd.DataFrame(columns=['dist', 'ratio', 'error', 'src_tgt'])
    for src_col in transfer_ratio['row_name']:
        tgt_cols = tr_cols.copy()

        inter_dist = ft_inter_dists[ft_inter_dists['row_name']==src_col][tgt_cols].to_numpy(dtype=np.float32).flatten()
        pef_ratio = transfer_ratio[transfer_ratio['row_name']==src_col][tgt_cols].to_numpy().flatten()
        pef_error = transfer_error[transfer_error['row_name']==src_col][tgt_cols].to_numpy().flatten()

        spec_src_stats = pd.DataFrame(columns=['dist', 'ratio', 'src_tgt'])
        spec_src_stats['dist'] = inter_dist
        spec_src_stats['ratio'] = pef_ratio
        spec_src_stats['error'] = pef_error
        spec_src_stats['src_tgt'] = [f'{src_col}→{abbr_dict[tgt_col]}' for tgt_col in tgt_cols] # 缩写

        transfer_stats = pd.concat([transfer_stats, spec_src_stats], axis='rows')

    return transfer_stats


def fitting_plotter(transfer_stats, title_):
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 100
    sns.set_palette("muted")

    # Linear regression fitting
    spr, p = spearmanr(transfer_stats['dist'], transfer_stats['ratio'])
    # Scatter plot each points
    fgrid = sns.lmplot(x="dist", y="ratio", data=transfer_stats, height=10, aspect=10/9, fit_reg=False,
                        legend=False, scatter_kws={"s": 150}, hue='tgt', palette='Set1')
    ax = sns.regplot(x="dist", y="ratio", data=transfer_stats, scatter_kws={"zorder":-1},
                    line_kws={'label': 'ρ: {:.3f}\np-value: {:.3f}'.format(spr, p)})
    # Annotate point
    texts = []
    for i, point in transfer_stats.iterrows():
        texts.append(plt.text(point['dist'], point['ratio'], point['src_tgt'], fontdict={'size': 16}))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='grey', lw=1.5))

    # Add error bar
    plt.errorbar(transfer_stats['dist'], transfer_stats['ratio'], yerr=transfer_stats['error'], fmt='', ls='none',
                ecolor='grey', elinewidth=4, capsize=6)
    # Add bbox via spine drawing
    for loc, spine in ax.spines.items(): # spine of left, right, bottom and top
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    # Label axis and show legend
    ax.set_xlabel("Distance of dataset finetuned representation", size=22)
    ax.set_ylabel("Relative change in test performance (%)", size=22)
    ax.tick_params(axis='both', labelsize=18)

    plt.legend(loc='lower left', prop={'size': 18}, ncol=1)
    plt.suptitle(f"{title_} Spearman ρ={spr:.3f}, p={p:.3f}", size=26, y=1.03)

    fig = plt.gcf()
    return fig


def set_seed(seed):
    np.random.seed(seed) # fix random seed to reproduce results
    torch.manual_seed(seed)         # Current CPU
    torch.cuda.manual_seed(seed)    # Current GPU
