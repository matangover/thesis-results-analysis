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
from pathlib import Path
import soundfile
import pandas
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio
sns.set(palette='Set2', font_scale=1.3)

# %%
# %load_ext autoreload
# %autoreload 1
# %aimport utils
import utils as u

# %%
nmf_results = u.read_evaluation_results('nmf_evaluation')


# %%
def get_results(dataset, config_name):
    return nmf_results.query(f'evaluation_dataset == "{dataset}" & model == "{config_name}"')


# %%
def show_results(dataset, config_name, metric='sdr', **kw):
    sns.catplot(
        data=get_results(dataset, config_name),
        y=metric,
        x='source',
        kind='box',
        showfliers=False
    )


# %%
sns.stripplot(nmf_results.source, nmf_results.sdr)

# %%
sns.catplot(
    data=get_results('chorales_synth_v5', '001'),
    y='sdr',
    x='source',
    kind='box',
    showfliers=False
)

# %%
sns.catplot(
    data=nmf_results.query('evaluation_dataset=="chorales_synth_v6"'),
    y='sdr',
    x='source',
    kind='box',
    showfliers=False
)

# %%
nmf_results.query('evaluation_dataset=="chorales_synth_v6"').sdr.mean()

# %%
show_results('chorales_synth_v6', '002')

# %%
get_results('chorales_synth_v6', '002').query('source=="soprano"').sdr.mean()

# %%
show_results('chorales_synth_v5', '003')

# %%
show_results('chorales_synth_v5', '002')

# %%
get_results('chorales_synth_v5', '001').query('source=="soprano"').sdr.mean()

# %%
nmf_results.model.unique()

# %%
sns.catplot(
    data=nmf_results.query('model in ["A", "B", "C", "D"] & evaluation_dataset == "chorales_synth_v5"').rename(
        columns={
            'sdr': 'SDR',
            'model': 'experiment',
            'evaluation_dataset': 'dataset'
        }
    ).replace({
        'dataset': {'chorales_synth_v5': 'normal', 'chorales_synth_v6': 'higher-variability'}
    }),
    y='SDR',
    x='experiment',
    order=['A', 'B', 'C', 'D'],
    dodge=True,
    kind='box',
    fliersize=1,
    col='source',
    sharex=False,
    aspect=0.5,
).set(ylim=(-8, 25)).set_titles('{col_name}')
plt.savefig('figures/nmf-results-normal.pdf', bbox_inches='tight')

# %%
sns.catplot(
    data=nmf_results.query('model in ["A", "B", "C", "D"] & evaluation_dataset == "chorales_synth_v6"').rename(
        columns={
            'sdr': 'SDR',
            'model': 'experiment',
            'evaluation_dataset': 'dataset'
        }
    ).replace({
        'dataset': {'chorales_synth_v5': 'normal', 'chorales_synth_v6': 'higher-variability'}
    }),
    y='SDR',
    x='experiment',
    order=['A', 'B', 'C', 'D'],
    dodge=True,
    kind='box',
    fliersize=1,
    col='source',
    sharex=False,
    aspect=0.5,
).set(ylim=(-8, 25)).set_titles('{col_name}')
plt.savefig('figures/nmf-results-high-variability.pdf', bbox_inches='tight')

# %%
sns.catplot(
    data=nmf_results.query('model in ["A", "B", "C", "D"]').rename(
        columns={
            'sdr': 'SDR',
            'model': 'experiment',
            'evaluation_dataset': 'dataset'
        }
    ).replace({
        'dataset': {'chorales_synth_v5': 'normal', 'chorales_synth_v6': 'higher-variability'}
    }),
    y='SDR',
    x='experiment',
    dodge=True,
    order=['A', 'B', 'C', 'D'],
    kind='box',
    fliersize=1,
    col='dataset',
    col_order=['normal', 'higher-variability'],
    sharex=False,
    aspect=0.7,
    width=0.5
).set_titles('{col_name}').set(ylim=(-20,25))
plt.savefig('figures/nmf-results-total.pdf', bbox_inches='tight')

# %% [markdown]
# # Worst frames

# %%
worst = nmf_results.query('model in ["A","B","C","D"]').sort_values('sdr').head(10)
worst[['model', 'evaluation_dataset', 'chorale', 'frame', 'source', 'sdr', 'sir', 'sar', 'isr']]

# %%
u.display_frame_from_row(worst.iloc[1], None, nmf=True, context_separate=True)

# %%
u.display_frame_from_row(worst.iloc[3], None, nmf=True, context_separate=True)

# %%
u.display_frame_from_row(worst.iloc[6], None, nmf=True, context_separate=True)
