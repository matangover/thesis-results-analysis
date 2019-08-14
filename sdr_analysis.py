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
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
sns.set(palette='Set2', font_scale=1.3)

# %%
# %load_ext autoreload
# %autoreload 1
# %aimport utils
import utils as u

# %%
results, unpivoted, models, datasets = u.load_results()

# %% {"jupyter": {"outputs_hidden": true}}
powers = u.get_frame_powers()

# %%
results_with_power = results.merge(powers, on=['dataset', 'chorale', 'source', 'frame'])

# %%
results_with_power_check_merge = results.merge(powers, on=['dataset', 'chorale', 'source', 'frame'], how='left')
results_with_power_check_merge[results_with_power_check_merge.power.isna()].dataset.unique()

# %%
len(results) == (len(results_with_power) + results_with_power_check_merge.power.isna().sum())

# %% [markdown]
# ## Energies

# %% {"jupyter": {"outputs_hidden": true}}
energies = u.get_frame_energies()

# %%
results_with_energy = results.merge(energies, on=['dataset', 'chorale', 'source', 'frame'])

# %%
results_with_energy_check_merge = results.merge(energies, on=['dataset', 'chorale', 'source', 'frame'], how='left')
results_with_energy_check_merge[results_with_energy_check_merge.energy.isna()].dataset.unique()

# %%
len(results) == (len(results_with_energy) + results_with_energy_check_merge.energy.isna().sum())

# %%
energies.energy.describe().to_frame().T

# %%
sns.distplot(energies.energy)

# %% [markdown]
# ## Distribution of reference frame power

# %%
sns.distplot(powers.power)

# %%
powers.power.describe().to_frame().T

# %%
plt.hist(powers.power, bins=20);

# %%
(powers.power<powers.power.quantile(0.1)).sum()

# %%
(powers.power < 0.00001).sum() / len(powers)

# %% [markdown]
# #### Reference frame power by source

# %%
sns.catplot(
    data=powers,
    x='source',
    y='power',
    kind='violin'
)

# %%
powers.sort_values('power', ascending=False)[:10]

# %%
u.display_reference_frame('chorales_synth_v6', 'soprano', 359, 59)

# %% [markdown]
# ### SDR against power

# %%
pmin, pmax = results_with_power.power.min(), results_with_power.power.max()
gap = (pmax - pmin) * 0.05
sns.scatterplot(
    data=results_with_power,
    y='sdr',
    x='power'
).set(xlim=(pmin - gap, pmax + gap));

# %%
plt.plot(results_with_power.power, results_with_power.sdr, ',')
plt.xlim(pmin - gap, pmax + gap)

# %%
# plt.figure(figsize=(8,6))
fig = sns.jointplot(
    data=u.rename_for_display(results_with_power),
    y='SDR',
    x='power',
    joint_kws={'marker': '.'},
    xlim=(pmin - gap, pmax + gap),
    height=8
)
fig.savefig('figures/sdr_against_power.pdf')

# %%
emin, emax = results_with_energy.energy.min(), results_with_energy.energy.max()
gap = (emax - emin) * 0.05

#plt.figure(figsize=(8,6), dpi=1200)
plt.rcParams['figure.dpi'] = 600
fig = sns.jointplot(
    data=u.rename_for_display(results_with_energy),
    y='SDR (dB)',
    x='energy',
    joint_kws={'marker': '.'},
    xlim=(emin - gap, emax + gap),
    height=8,
    rasterized=True
)
fig.savefig('figures/sdr_against_energy.pdf', bbox_inches='tight')
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ### Log scale

# %%
emin, emax = results_with_energy.energy.min(), results_with_energy.energy.max()
gap = (emax - emin) * 0.05

#plt.figure(figsize=(8,6), dpi=1200)
#plt.rcParams['figure.dpi'] = 600

results_with_energy_log = results_with_energy.assign(energy_log=10*np.log10(results_with_energy.energy/emax))
fig = sns.jointplot(
    data=u.rename_for_display(results_with_energy_log),
    y='SDR (dB)',
    x='energy_log',
    joint_kws={'marker': '.'},
#    xlim=(emin - gap, emax + gap),
    height=8,
    rasterized=True
)
#fig.ax_joint.set_xscale('log')
fig.savefig('figures/sdr_against_energy_log_scale.pdf', bbox_inches='tight')
#plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## Max abs of sources in dataset
#
# Is really low, oops.

# %% {"jupyter": {"outputs_hidden": true}}
max_abs = u.get_max_abs()

# %%
max_abs

# %% [markdown]
# ### Worst frame SDR

# %%
results.loc[results.sdr.idxmin()][['evaluation_dataset', 'model', 'source', 'chorale', 'frame', 'sdr']].to_frame().T

# %%
results.sort_values('sdr').head(10)[['evaluation_dataset', 'model', 'source', 'chorale', 'frame', 'sdr']]

# %%
reference, estimate = u.load_frame('chorales_synth_v5', 'alto', 358, 39, '006', models)

# %%
Audio(reference, rate=44100, normalize=False)

# %%
np.unique(reference, return_counts=True)

# %%
u.get_snr(reference, estimate)

# %%
u.get_snr(reference, estimate, 30)

# %% [markdown]
# ## Frames that are quiet but good

# %%
most_quiet = results_with_power[results_with_power.power < results_with_power.power.quantile(0.001)]

# %%
len(most_quiet)

# %%
quiet_and_good_sdr = most_quiet.sort_values('sdr', ascending=False)
quiet_and_good_sdr.head()[['power', 'model', 'sdr', 'evaluation_dataset', 'chorale', 'frame', 'source']]

# %%
u.display_frame_from_row(quiet_and_good_sdr.iloc[0], models, context=0)

# %%
powers.power.value_counts().sort_index().head(20)

# %% [markdown]
# ## Only experiment 4

# %%
experiment4_models = ['036', '057', '058', '059']
experiment4_results = results_with_power[results_with_power.model.isin(experiment4_models)]

# %%
plt.plot(experiment4_results.power, experiment4_results.sdr, '.')
plt.xlim(pmin - gap, pmax + gap)

# %%
experiment4_results_lowest = experiment4_results.assign(lowest=(experiment4_results.power < experiment4_results.power.quantile(0.03)))
sns.scatterplot(
    'power', 'sdr',
    hue='lowest',
    data=experiment4_results_lowest,
)
plt.xlim(pmin - gap, pmax + gap)

# %%
lowest_loudest = experiment4_results_lowest[experiment4_results_lowest.lowest].sort_values('power', ascending=True)
lowest_loudest.head()

# %%
u.display_frame_from_row(lowest_loudest.iloc[1], models, context=0)

# %% [markdown]
# ### Single chorale, single source

# %%
model = '036' # soprano experiment4
chorale = 358
res = results_with_power.query(f'model == {model!r} & chorale == {chorale}')
plt.plot(res.power, res.sdr, '.');

# %%
res.sort_values('power').iloc[0][['evaluation_dataset', 'chorale', 'source', 'frame', 'power']]

# %%
u.display_reference_frame('chorales_synth_v6', 'soprano', 358, 4)

# %%
u.display_frame_from_row(res.sort_values('power').iloc[0], models, context_separate=True)

# %%
experiment4_results_lowest = experiment4_results.assign(lowest=(experiment4_results.power < experiment4_results.power.quantile(0.03)))
sns.scatterplot(
    'power', 'sdr',
    hue='lowest',
    data=experiment4_results_lowest,
)
plt.xlim(pmin - gap, pmax + gap)

# %%
lowest_loudest = experiment4_results_lowest[experiment4_results_lowest.lowest].sort_values('power', ascending=True)
lowest_loudest.head()

# %%
u.display_frame_from_row(lowest_loudest.iloc[1], models, context=0)

# %% [markdown]
# # SNR Regularization

# %%
sdr_358 = u.evaluate_sdr_chorale('chorales_synth_v6', '036', 358, 'soprano', models, regularization=0.0001)

# %%
sdr_358.head()

# %% [markdown]
# #### Regularization boosts up SNR for this frame

# %%
u.display_frame('chorales_synth_v6', ['soprano'], 358, 4, '036', models, context=1, context_separate=True)

# %% [markdown]
# #### But not for this one!

# %%
u.display_frame('chorales_synth_v6', ['soprano'], 358, 18, '036', models, context=1, context_separate=True)

# %%
sns.jointplot(data=sdr_358, x='sdr_bsseval', y='sdr_reg');

# %%
sdr_358_with_keys = sdr_358.assign(chorale=358, model='036', source='soprano')
results_358_with_sdr = results_with_power.merge(sdr_358_with_keys, on=['chorale', 'model', 'source', 'frame'])
assert len(results_358_with_sdr) == results[(results.chorale==358) & (results.evaluation_dataset=='chorales_synth_v6')].frame.nunique()

# %%
results_358_with_sdr_melted = results_358_with_sdr.melt(id_vars='power', value_vars=['sdr', 'sdr_reg'], var_name='metric')

# %%
sns.scatterplot(
    x='power',
    y='value',
    hue='metric',
    alpha=0.7,
    palette=['blue', 'yellow'],
    data=results_358_with_sdr_melted
).set(xlim=(-0.00001, 0.0001));

# %%
sns.scatterplot(data=results_358_with_sdr, x='sdr_bsseval', y='sdr_reg', hue='power');

# %%
plt.plot(results_358_with_sdr.power, results_358_with_sdr.sdr_reg, '.');

# %% [markdown]
# ### Same for alto

# %%
sdr_358_alto = u.evaluate_sdr_chorale('chorales_synth_v6', '057', 358, 'alto', models, regularization=0.001)

# %%
sdr_358_alto

# %%
u.display_frame('chorales_synth_v6', ['alto'], 358, 36, '057', models, context=1, context_separate=True)

# %% [markdown]
# ### Evaluate effect of regularization on all chorales in dataset, Experiment 4

# %%
res = u.evaluate_sdr_models('chorales_synth_v6', experiment4_models, models, 0.0001)

# %%
res.head()

# %%
res=res.astype({'chorale': int})

# %%
experiment4_results_with_sdr_reg = results_with_power.merge(res, on=['chorale', 'model', 'source', 'frame'])

# %%
sns.scatterplot(data=experiment4_results_with_sdr_reg, x='sdr_bsseval', y='sdr_reg', hue='power');

# %%
u.display_frame('chorales_synth_v6', ['soprano'], 345, 9, '036', models, context=1, context_separate=True)

# %%
plt.figure(figsize=(8,5))
sns.scatterplot(y=experiment4_results_with_sdr_reg.sdr_reg-experiment4_results_with_sdr_reg.sdr_bsseval, x=experiment4_results_with_sdr_reg.power)
plt.xlim(-0.000015, 0.00025);
plt.ylabel(r'SDR$_{\rm reg}$ - SDR')
plt.savefig('figures/sdr-reg-vs-sdr-old.pdf')

# %%
fig = sns.jointplot(
    data=u.rename_for_display(experiment4_results_with_sdr_reg),
    y='sdr_reg',
    x='power',
    joint_kws={'marker': '.'},
    xlim=(pmin - gap, pmax + gap),
    height=8
)

# %%
experiment4_models = ['036', '057', '058', '059']
experiment4_results_en = results_with_energy[results_with_energy.model.isin(experiment4_models)]

# %% [markdown]
# ## Energies - SDR vs SDR reg

# %%
plt.figure(figsize=(8,5))
experiment4_sdr_reg = u.evaluate_sdr_models('chorales_synth_v6', experiment4_models, models, 0.0001)
experiment4_results_with_sdr_reg = results_with_energy.merge(experiment4_sdr_reg, on=['chorale', 'model', 'source', 'frame'])

# %%
sns.scatterplot(y=experiment4_results_with_sdr_reg.sdr_reg-experiment4_results_with_sdr_reg.sdr_bsseval, x=experiment4_results_with_sdr_reg.energy)
plt.ylabel(r'SDR$_{\rm reg}$ - SDR  (dB)')
plt.savefig('figures/sdr-reg-vs-sdr.pdf', bbox_inches='tight')

# %% [markdown]
# # Check specific frames with 'wrong' SDR

# %%
u.display_frame('chorales_synth_v6', ['soprano'], 337, 21, '036', models, context=1, context_separate=True)
