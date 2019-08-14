# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
# ---

# %%
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML
import tabulate

sns.set(palette='Set2', font_scale=1.3)

# %%
# %load_ext autoreload

# %%
# %autoreload 1
# %aimport utils

# %%
import utils as u

# %%
results, unpivoted, models, datasets = u.load_results()

# %% [markdown]
# # Experiment 1: Soprano-Bass separation
#
# Model 308440

# %%
experiment1 = results[results.model == '308440']

# %%
import matplotlib
l=matplotlib.cm.get_cmap(name='Set2')

# %%
l.colors[0]

# %%
sns.catplot(
    data=u.rename_for_display(experiment1),
    y='SDR',
    x='source',
    kind='box',
    showfliers=False,
    order=['soprano', 'bass'],
    width=0.3,
    color=l.colors[0],
    aspect=1.2
#    fliersize=2,
)
plt.xlabel(None)
plt.savefig('figures/sb-sdr.pdf', bbox_inches='tight')

# %%
metrics_table = experiment1.groupby('source').agg({'sdr': ['median', 'mean', 'std', 'min', 'max']}).dropna()
metrics_table

# %%
print(u.markdown_table(metrics_table['sdr']))

# %% [markdown] {"toc-hr-collapsed": true}
# # Experiment 2: All Four Voices
#
# Model 914434

# %%
experiment2 = results[results.model == '914434']

# %%
sns.catplot(
    data=u.rename_for_display(experiment2),
    y='SDR',
    x='source',
    kind='box',
    showfliers=False,
    showmeans=True,
    meanline=True,
    meanprops={'color': 'black'},
    color=l.colors[0],
    width=0.4,
    aspect=1.3,
#    fliersize=2,
#    whis=np.inf
)
plt.xlabel(None)
plt.savefig('figures/satb-sdr.pdf', bbox_inches='tight')

# %%
metrics_table = experiment2.groupby('source').agg({'sdr': ['median', 'mean', 'std', 'min', 'max']})
metrics_table

# %%
print(u.markdown_table(metrics_table['sdr']))

# %% [markdown]
# #### Examine worst frames

# %%
worst = experiment2.groupby(['source', 'chorale', 'frame']).sdr.mean().to_frame().sort_values('sdr')
ten_worst = worst.head(10).reset_index()
ten_worst

# %%
print(u.markdown_table(ten_worst, showindex=False))

# %%
for i, row in enumerate(worst.head(10).reset_index().itertuples()):
    display(HTML(f'<h4>{i}. chorale: {row.chorale}, frame: {row.frame}, {row.source}. SDR: {row.sdr} </h4>'))
    u.display_frame('chorales_synth_v5', ['alto', 'tenor'], row.chorale, row.frame, '914434', models)

# Analysis:
# 0-1. Silent
# 2-9. Voice crossing alto-tenor

# %% [markdown]
# #### Experiment 1 vs Experiment 2

# %%
experiment1_vs_2 = pandas.concat([experiment1, experiment2]).query('source in ["soprano", "bass"]')
experiment1_vs_2['experiment'] = experiment1_vs_2.model.map({'308440': 'Experiment 1', '914434': 'Experiment 2'})

# %%
sns.catplot(
    data=u.rename_for_display(experiment1_vs_2),
    y='SDR',
    x='source',
    kind='box',
    showfliers=False,
    hue='experiment',
    aspect=1.4,
    width=0.4,
    order=['soprano', 'bass'],
#    fliersize=2,
#    whis=np.inf
)
plt.xlabel(None)
plt.savefig('figures/experiment1-vs-experiment2.pdf', bbox_inches='tight')

# %% [markdown]
# # Experiment 3
#
# Models: 013 (soprano), 007 (alto), 009 (tenor), 011 (bass).

# %%
experiment3_models = ['013', '007', '009', '011']
experiment3 = results[results.model.isin(experiment3_models)]

# %%
sns.boxplot(
    data=experiment3,
    y='sdr',
    x='source',
    showfliers=False,
#    hue='experiment'
);

# %%
experiment2_vs_3 = pandas.concat([
    experiment2.assign(experiment='Experiment 2'),
    experiment3.assign(experiment='Experiment 3')
])

# %%
fig = sns.catplot(
    data=u.rename_for_display(experiment2_vs_3),
    y='SDR',
    x='source',
    showfliers=False,
    hue='experiment',
    kind='box',
    aspect=1.8,
    width=0.55,
)
plt.xlabel(None)
fig.savefig('figures/experiment2-vs-experiment3.pdf', bbox_inches='tight')

# %%
metrics_table = experiment3.groupby('source').agg({'sdr': ['median', 'mean', 'std', 'min', 'max']})
metrics_table

# %% [markdown]
# ### Examine worst frames - Experiment 3

# %% {"jupyter": {"outputs_hidden": true}}
worst = experiment3.groupby(['source', 'model', 'chorale', 'frame']).sdr.mean().to_frame().sort_values('sdr')
ten_worst = worst.head(10).reset_index()
ten_worst

# %% {"jupyter": {"outputs_hidden": true}}
for i, row in enumerate(ten_worst.head(10).reset_index().itertuples()):
    display(HTML(f'<h4>{i}. chorale: {row.chorale}, frame: {row.frame}, {row.source}, model: {row.model}. SDR: {row.sdr} </h4>'))
    u.display_frame('chorales_synth_v5', [row.source], row.chorale, row.frame, row.model, models)

# %% [markdown]
# # Experiment 4
#
# Models: 036 (soprano), 057 (alto), 058 (tenor), 059 (bass).

# %%
experiment4_models = ['036', '057', '058', '059']
experiment4 = results[results.model.isin(experiment4_models)]

# %%
sns.boxplot(
    data=experiment4,
    y='sdr',
    x='source',
    showfliers=False,
#    hue='experiment'
);

# %%
experiment3_vs_4 = pandas.concat([
    experiment3.assign(experiment='Experiment 3'),
    experiment4.assign(experiment='Experiment 4')
])
fig = sns.catplot(
    data=u.rename_for_display(experiment3_vs_4),
    y='SDR',
    x='source',
    showfliers=False,
#    fliersize=1,
    hue='experiment',
    kind='box',
    aspect=1.8,
    width=0.55,
#    height=7
);
plt.xlabel(None)
fig.savefig('figures/experiment3-vs-experiment4.pdf', bbox_inches='tight')

# %%
by_frame = experiment3_vs_4.groupby(['source', 'experiment', 'model', 'chorale', 'frame']).sdr.mean().to_frame().reset_index()
fig = sns.catplot(
    data=u.rename_for_display(by_frame),
    y='SDR',
    x='source',
    hue='experiment',
    dodge=True,
    aspect=2
)
plt.xlabel(None)
fig.savefig('figures/experiment3-vs-experiment4-strip.pdf', bbox_inches='tight')

# %% [markdown]
# ### Examine worst frames - Experiment 4

# %%
experiment4_by_sdr = experiment4.sort_values('sdr')
ten_worst = experiment4_by_sdr.head(10).reset_index()
ten_worst

# %% {"jupyter": {"outputs_hidden": true}}
for i, row in enumerate(ten_worst.itertuples()):
    display(HTML(f'<h4>{i}. chorale: {row.chorale}, frame: {row.frame}, {row.source}, model: {row.model}. SDR: {row.sdr} </h4>'))
    u.display_frame(row.evaluation_dataset, [row.source], row.chorale, row.frame, row.model, models, context_separate=True)

# %%
sns.distplot(experiment4.sdr[experiment4.sdr < 0], kde=False) #query('-10 < sdr < -5'))

# %%
for i, row in enumerate(experiment4_by_sdr.query('-20 < sdr').head(20).itertuples()):
    display(HTML(f'<h4>{i}. chorale: {row.chorale}, frame: {row.frame}, {row.source}, model: {row.model}. SDR: {row.sdr:.2f} </h4>'))
    u.display_frame(row.evaluation_dataset, [row.source], row.chorale, row.frame, row.model, models, context_separate=True)

# %%
worst = experiment4.groupby(['source', 'model', 'chorale', 'frame'], observed=True).sdr.mean().to_frame().sort_values('sdr')
sns.distplot(worst.sdr, kde=False);

# %%
for i, row in enumerate(worst.itertuples()):
    display(HTML(f'<h4>{i}. chorale: {row.chorale}, frame: {row.frame}, {row.source}, model: {row.model}. SDR: {row.sdr} </h4>'))
    u.display_frame('chorales_synth_v6', [row.source], row.chorale, row.frame, row.model, models, context_separate=True)

# %% [markdown]
# # Comparison: Experiments 1-4

# %%
experiments1_to_4 = pandas.concat([
    experiment1.assign(experiment='Experiment 1'),
    experiment2.assign(experiment='Experiment 2'),
    experiment3.assign(experiment='Experiment 3'),
    experiment4.assign(experiment='Experiment 4')
])
fig = sns.catplot(
    data=u.rename_for_display(experiments1_to_4),
    y='SDR',
    x='source',
#    showfliers=True,
    fliersize=1.2,
    hue='experiment',
    kind='box',
    aspect=1.6,
    width=0.7,
    height=8,
    legend_out=False
);
plt.xlabel(None)
fig.savefig('figures/experiments-1-to-4.pdf', bbox_inches='tight')
