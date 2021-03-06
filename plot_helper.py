import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from features_helper import *

# Init the style for all the plots
def init_style():
    sns.set_style("whitegrid")
    params = {'legend.fontsize': 15,
          'legend.title_fontsize': 15,
          'figure.figsize': (10, 5),
          'axes.labelsize': 15,
          'axes.titlesize': 20,
          'figure.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
    plt.rcParams.update(params)

# Compute the bootstrap standard error of given data
def bootstrap_standard_error(data, nb_samples, nb_iteration):
    means = []
    for i in range(nb_iteration):
        # Take randomly n_samples samples with replacement
        samples = resample(data, replace=True, n_samples=nb_samples)
        means.append(np.mean(samples)) # Compute the mean of the distribution
    return np.std(means) # return the standard error

# Draw an imbalance plot similar to the one we can observe in the paper for a selected property
def draw_imbalance_plot(ax_imb, ax_player, data, property_name, imb_lim, player_lim, legend=False, mode='season'):
    
    # Compute the mean statistics
    stats = data.groupby(['player_type', 'betrayal'])[property_name].agg(['mean', 'count'])
    stats['standard_error'] = data.groupby(['player_type', 'betrayal'])[property_name].apply(lambda grp: bootstrap_standard_error(grp, grp.count(), 1000))
    stats = stats.reset_index()
    stats = stats.sort_values(by='player_type', ascending=True)

    # Compute the imbalance
    if mode == 'season': # If the dataset correspond to seaons features, merge them by season
        on = ['idx', 'season', 'betrayal']
    else: # Otherwise, merge them by friendship
        on = ['idx', 'betrayal']
    imbalance = merge_player_features(data, on=on)
    imbalance['imbalance'] = imbalance[property_name + '_betrayer'] - imbalance[property_name + '_victim']

    # Compute the imbalance statistics
    imbalance_stats = imbalance.groupby('betrayal')['imbalance'].agg(['mean', 'count'])
    imbalance_stats['standard_error'] = imbalance.groupby('betrayal')['imbalance'].apply(lambda grp: bootstrap_standard_error(grp, grp.count(), 1000))
    imbalance_stats = imbalance_stats.reset_index()
    
    # Size of the bars in the plot
    barwidth = 0.2
    
    # Load the data for the plot
    imb_false = imbalance_stats[imbalance_stats['betrayal'] == False]
    imb_true = imbalance_stats[imbalance_stats['betrayal'] == True]
    
    player_false = stats[stats['betrayal'] == False]
    player_true = stats[stats['betrayal'] == True]
    
    # Draw the imbalance plot
    ax_imb.bar(-barwidth, imb_false['mean'], 2 * barwidth)
    ax_imb.bar(barwidth, imb_true['mean'], 2 * barwidth)
    ax_imb.errorbar(x=-barwidth, y=imb_false['mean'], yerr=imb_false['standard_error'], fmt='none', ecolor='black', elinewidth=2, capsize=10, capthick=2)
    ax_imb.errorbar(x=barwidth, y=imb_true['mean'], yerr=imb_true['standard_error'], fmt='none', ecolor='black', elinewidth=2, capsize=10, capthick=2)
    ax_imb.set(ylim=imb_lim)
    ax_imb.set(xlabel='imbalance', ylabel=None)
    ax_imb.tick_params(bottom=False, labelbottom=False)

    x = np.arange(2)
    
    # Draw the mean plot
    bar_false = ax_player.bar(x-barwidth, player_false['mean'], 2 * barwidth)
    bar_true = ax_player.bar(x+barwidth, player_true['mean'], 2 * barwidth)
    ax_player.errorbar(x=x-barwidth, y=player_false['mean'], yerr=player_false['standard_error'], fmt='none', ecolor='black', elinewidth=2, capsize=10, capthick=2)
    ax_player.errorbar(x=x+barwidth, y=player_true['mean'], yerr=player_true['standard_error'], fmt='none', ecolor='black', elinewidth=2, capsize=10, capthick=2)
    ax_player.set(ylim=player_lim)
    ax_player.set(xlabel=None, ylabel=None)
    ax_player.set_xticks(x)
    ax_player.set_xticklabels(labels=['(potential) betrayer', '(potential) victim'])
    # Display the legend if required
    if legend:
        ax_player.legend([bar_false, bar_true], ['No betrayal', 'Betrayal'], loc='upper right', frameon=False)

# Draw a heatmap plot corresponding the cross-corelation of the features
def corr_plot(corr, title):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    sns.heatmap(corr, ax=ax, vmin=0, vmax=1, center=0, square=True, cbar_kws={"shrink": 0.7})
    
    ax.set_xticks(np.arange(len(corr.columns)) + 0.5)
    ax.set_yticks(np.arange(len(corr.columns)) + 0.5)
    ax.set_xticklabels(corr.columns, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    ax.set_title(title)

# Draw a violin plot corresponding to the correlation with the betrayal of specific groups of features
def betrayal_corr_plot(betrayal_corr, y_col, y_name, title):
    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    sns.violinplot(ax=ax, data=betrayal_corr, y=y_col, x='betrayal_correlation', inner="box", color="#999999", cut=0, width=1, linewidth=3)
    ax.set(xlabel='Betrayal correlation', ylabel=y_name)
    
    ax.set_title(title)

