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
# # Experiment 6: Midi-norm, single voice (SATB)
#
# 012 (soprano), 006 (alto), 008 (tenor), 010

# %%
experiment6_models = ['012', '006', '008', '010']
experiment6 = results[results.model.isin(experiment6_models)]

# %%
sns.boxplot(
    data=experiment6,
    y='sdr',
    x='source',
    showfliers=False
#    fliersize=2,
#    whis=np.inf
)
plt.savefig('figures/experiment6.pdf')

# %% [markdown]
# ### Experiment 6 vs Experiment 3 (single voice SATB, no score)

# %%
experiment3_models = ['013', '007', '009', '011']
experiment3 = results[results.model.isin(experiment3_models)]

# %%
experiment6_vs_3 = pandas.concat([
    experiment6.assign(experiment='Experiment 6'),
    experiment3.assign(experiment='Experiment 3')
])

# %%
fig = sns.catplot(
    data=u.rename_for_display(experiment6_vs_3),
    y='SDR',
    x='source',
    showfliers=False,
    hue='experiment',
    kind='box',
    aspect=1.7,
    width=0.55,
    hue_order=['Experiment 3', 'Experiment 6']
);
plt.xlabel(None)
fig.savefig('figures/experiment6-vs-experiment3.pdf', bbox_inches='tight')

# %%
experiment6[experiment6.source=='soprano'].sdr.median()

# %%
experiment6_vs_3.query('chorale == 358 & frame == 46 & source == "alto"')[['experiment', 'model', 'sdr']]

# %%
u.display_frame('chorales_synth_v5', ['alto'], 358, 46, '006', models, context=1, context_separate=True)

# %% [markdown]
# #### Voice crossing frames -- did they improve between Experiment 2 / 3 to 6?

# %%
crossings_df = pandas.DataFrame([
        ['alto', 358, 46],
        ['alto', 341, 19],
        ['tenor', 367, 3],
        ['tenor', 367, 4],
        ['alto', 366, 20],
        ['tenor', 358, 45],
        ['tenor', 358, 48],
        ['tenor', 358, 44],
    ],
    columns=['source', 'chorale', 'frame']
)

# %%
experiment2_models = ['914434']
experiment2_crossings = results[results.model.isin(experiment2_models)].merge(crossings_df)
experiment3_crossings = results[results.model.isin(experiment3_models)].merge(crossings_df)
experiment6_crossings = results[results.model.isin(experiment6_models)].merge(crossings_df)

# %%
crossings_comparison = crossings_df.assign(
    ex2=experiment2_crossings.sdr,
    ex3=experiment3_crossings.sdr,
    ex6=experiment6_crossings.sdr,
).rename(columns={'ex2': 'Ex. 2 SDR', 'ex3': 'Ex. 3 SDR', 'ex6': 'Ex. 6 SDR'})
crossings_comparison

# %%
print(u.markdown_table(crossings_comparison, showindex=False))

# %% [markdown]
# # Experiment 7 - multi-source SATB
#
# Model 016

# %%
fig = sns.catplot(
    data=u.experiment_results(7, results),
    y='sdr',
    x='source',
    showfliers=False,
#    hue='experiment',
    kind='box',
    aspect=1.5,
);
fig.savefig('figures/experiment7.pdf')

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiments_results([3, 6, 7], results)),
    y='SDR',
    x='source',
    showfliers=False,
    hue='experiment',
    kind='box',
    aspect=1.9,
    width=0.55
);
plt.xlabel(None)
fig.savefig('figures/experiment7-vs-3-and-6.pdf', bbox_inches='tight')

# %% [markdown]
# # Experiment 8
# 12 models: 78-86, 90-92

# %%
u.experiment_results(8, results).groupby(['score_concat', 'score_type']).model.unique().reset_index().pivot('score_concat', 'score_type').model

# %%
fig = sns.catplot(
    data=u.experiment_results(8, results),
    y='sdr',
    x='source',
    hue='score_concat',
    col='score_type',
    showfliers=False,
    kind='box',
    aspect=1.5,
);

# %%
fig = sns.catplot(
    data=u.experiment_results(8, results),
    y='sdr',
    x='source',
    col='score_concat',
    hue='score_type',
    showfliers=False,
    kind='box',
    aspect=1.5,
);

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiments_results([2, 8], results)),
    y='SDR',
    x='Score Type',
    row='source',
    hue='Conditioning Location',
    showfliers=False,
    order=['normalized pitch', 'pure tone', 'pitch and amplitude', 'piano roll'],
    hue_order=['input', 'output', 'input-output'],
    kind='box',
    sharex=False,
    aspect=1.9,
    width=0.7
);
plt.xlabel(None)
fig.savefig('figures/experiment8.pdf', bbox_inches='tight')

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiments_results([2, 8], results)),
    y='sdr',
    x='Score Type',
    col='source',
    hue='Conditioning Location',
    kind='strip',
    dodge=True,
    col_wrap=2,
    aspect=1.8,
    sharex=False,
);

# %%
fig = sns.catplot(
    data=u.experiment_results(8, results),
    y='sdr',
    x='score_type',
#    col='score_concat',
    hue='score_concat',
#    showfliers=False,
#    hue='experiment',
#    kind='box',
    aspect=1.5,
#    marker='source',
    dodge=True
);
#fig.savefig('figures/experiment7.pdf')

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiment_results(8, results)),
    y='sdr',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    kind='box',
    aspect=1.5,
    legend_out=True
)
#fig._legend.set_title('Conditioning location')
#fig.savefig('figures/experiment7.pdf')

# %% [markdown]
# ### Experiment 8 vs. Experiment 2 (NOT FAIR -- Experiment 2 is v5)

# %%
ex8vs2 = u.experiments_results([8, 2], results).fillna({'score_type': 'no score', 'score_concat': 'no score'})

# %%
ex8vs2.groupby(['score_concat', 'score_type']).model.unique().reset_index().pivot('score_concat', 'score_type').model

# %%
fig = sns.catplot(
    data=ex8vs2,
    y='sdr',
    x='score_type',
    col='source',
    hue='score_concat',
    showfliers=False,
    kind='box',
    col_wrap=2,
    aspect=1.8,
);

# %%
fig = sns.catplot(
    data=ex8vs2,
    y='sdr',
    x='score_type',
    hue='score_concat',
    aspect=1.5,
    dodge=True
);
#fig.savefig('figures/experiment7.pdf')

# %%
fig = sns.catplot(
    data=ex8vs2,
    y='sdr',
    x='score_type',
    hue='score_concat',
    aspect=1.5,
    dodge=True,
    hue_order=['in', 'out', 'in-out', 'no score'],
    col='source',
    col_wrap=2
);
#fig.savefig('figures/experiment7.pdf')

# %%
fig = sns.catplot(
    data=u.rename_for_display(ex8vs2),
    y='sdr',
    x='Score Type',
    hue='Conditioning Location',
    kind='strip', dodge=True,
    aspect=1.8,
    legend_out=True
)
#fig._legend.set_title('Conditioning location')
#fig.savefig('figures/experiment7.pdf')

# %%
ex8 = u.experiment_results(8, results)
ex8.sort_values('sdr').head(10)[['chorale', 'source', 'frame', 'model', 'sdr']]

# %%
u.display_frame('chorales_synth_v6', ['bass'], 364, 10, '079', models, context_separate=True)

# %% [markdown]
# # Experiment 9
# 12 models: 69-77, 87-89

# %%
u.experiment_results(9, results).groupby(['score_concat', 'score_type']).model.unique().reset_index().pivot('score_concat', 'score_type').model

# %%
u.experiment_results(9, results).source.unique()

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiment_results(9, results)),
    y='sdr',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    order=['normalized pitch', 'synthesized pure tone', 'pitch and amplitude', 'piano roll'],
    hue_order=['input', 'output', 'input-output'],
    kind='box',
    aspect=1.8,
    width=0.7,
);
fig.savefig('figures/experiment9.pdf')

# %% [markdown]
# ### Experiment 9 vs. no score
# Comparing to no-score tenor extraction from v6 (model 058)

# %%
ex9_noscore = results[results.model == '058'].fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 9 (no score)')
ex9_vs_noscore = pandas.concat([u.experiment_results(9, results), ex9_noscore])

# %%
ex9_vs_noscore.groupby(['score_concat', 'score_type']).model.unique().reset_index().pivot('score_concat', 'score_type').model

# %%
fig = sns.catplot(
    data=u.rename_for_display(ex9_vs_noscore),
    y='SDR',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    order=['normalized pitch', 'pure tone', 'pitch and amplitude', 'piano roll', 'no score'],
    hue_order=['input', 'output', 'input-output', 'no score'],
    kind='box',
    aspect=2,
)
fig.savefig('figures/experiment9-vs-noscore.pdf')

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiment_results(9, results)),
    y='SDR',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    order=['normalized pitch', 'pure tone', 'pitch and amplitude', 'piano roll'],#, 'no score'],
    hue_order=['input', 'output', 'input-output'],#, 'no score'],
    kind='box',
    aspect=2.2,
    width=0.6
)
plt.xlabel(None)
fig.savefig('figures/experiment9-vs-noscore.pdf')
print(plt.ylim())

# %%
import matplotlib

fig = sns.catplot(
    data=u.rename_for_display(ex9_noscore),
    y='SDR',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    kind='box',
    color='purple',
    aspect=0.5,
    width=0.5,
    saturation=100
)
plt.ylim((-5.974679340170534, 20.71880851271939))
plt.xlabel(None);
fig.savefig('figures/experiment9-noscore.pdf')

# %% [markdown]
# # Experiment 10
# 8 models: 062, 064, 066, 026, 022, 033, 060, 068

# %%
u.experiment_results(10, results).groupby(['score_concat', 'score_type']).model.unique().reset_index().pivot('score_concat', 'score_type').model

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiment_results(10, results)),
    y='SDR',
    x='Score Type',
    hue='Conditioning Location',
    row='source',
    showfliers=False,
    order=['normalized pitch', 'pure tone', 'pitch and amplitude', 'piano roll'],
    hue_order=['input', 'input-output'],
    kind='box',
    aspect=1.5,
    width=0.55,
    sharex=False
);
plt.xlabel(None)
fig.savefig('figures/experiment10.pdf', bbox_inches='tight')

# %%
fig = sns.catplot(
    data=u.rename_for_display(u.experiment_results(10, results)),
    y='SDR',
    x='Score Type',
    hue='Conditioning Location',
    showfliers=False,
    order=['normalized pitch', 'pure tone', 'pitch and amplitude', 'piano roll'],
    hue_order=['input', 'input-output'],
    kind='box',
    aspect=1.5,
    width=0.5,
    sharex=False
);
#fig.savefig('figures/experiment10_b.pdf')

# %%
ex10_by_median_sdr = u.experiment_results(10, results).groupby(['score_type', 'score_concat','model'], observed=True).sdr.median().sort_values(ascending=False).reset_index()
ex10_by_median_sdr

# %%
for source in u.all_sources:
    print(source)
    display(u.experiment_results(10, results).query(f'source=="{source}"').groupby(['score_type', 'score_concat','model'], observed=True).sdr.median().sort_values(ascending=False).reset_index())

# %%
print(
    u.markdown_table(
        u.rename_for_display(ex9_by_median_sdr.reset_index()).rename(columns={'sdr': 'Median SDR'}),
        showindex=False
    )
)

# %% [markdown]
# ### Bad frames: experiment 10

# %%
experiment10_by_sdr = u.experiment_results(10, results).sort_values('sdr')
experiment10_by_sdr.head()[['chorale', 'frame', 'source', 'model', 'sdr']]

# %% [markdown]
# Model outputs noise instead of silence:

# %%
u.display_frame_from_row(experiment10_by_sdr.iloc[0], models)

# %%
experiment10_by_sdr_best = u.experiment_results(10, results).sort_values('sdr', ascending=False)
experiment10_by_sdr_best.head()[['chorale', 'frame', 'source', 'model', 'sdr']]

# %%
by_sdr = u.experiment_results(10, results).sort_values('sdr')
u.display_frame_from_row(by_sdr.iloc[-50], models, context_separate=True)

# %%
experiment9_by_sdr.head()

# %% [markdown]
# # Comparison: Experiments 8-10 and 4

# %% [markdown]
# #### Best in Experiment 9

# %%
u.experiment_results(9, results).groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
ex9_by_median_sdr = u.experiment_results(9, results).groupby(['score_type', 'score_concat']).sdr.median().sort_values(ascending=False)
print(
    u.markdown_table(
        u.rename_for_display(ex9_by_median_sdr.reset_index()).rename(columns={'sdr': 'Median SDR'}),
        showindex=False
    )
)

# %% [markdown]
# #### Best in Experiment 8 (tenor)

# %%
u.experiment_results(8, results).query('source == "tenor"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %% [markdown]
# #### Best in Experiment 10 (tenor)

# %%
u.experiment_results(10, results).query('source == "tenor"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %% [markdown]
# #### Best in model 020 - no-score SATB extraction v6

# %%
results.query('model == "020" & source == "tenor"').sdr.median()

# %% [markdown]
# #### Best in model 058 - no-score tenor extraction v6

# %%
results.query('model == "058"').sdr.median()

# %% [markdown]
# #### Compare

# %%
ex9_noscore = results[results.model == '058'].fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 9 (no score)').assign(condition='No score: tenor')
ex8_tenor_best = u.experiment_results(8, results).query('source == "tenor" & model == "083"').assign(condition='Score: SATB')
ex8_noscore = results.query('model == "020" & source == "tenor"').fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 8 (no score)').assign(condition='No score: SATB')
ex9_best = u.experiment_results(9, results).query('model == "071"').assign(condition='Score: tenor')
ex10_best = u.experiment_results(10, results).query('model == "062"').assign(condition='Score: multi-source')

ex8910_compare = pandas.concat([ex9_best, ex9_noscore, ex8_noscore, ex8_tenor_best, ex10_best])

# %%
fig = sns.catplot(
    data=u.rename_for_display(ex8910_compare),
    y='SDR',
    x='condition',
    order=['Score: multi-source', 'Score: tenor', 'Score: SATB', 'No score: tenor', 'No score: SATB'],
     kind='box', showfliers=False, width=0.4,
    aspect=1.8,
)
plt.xlabel(None);
fig.savefig('figures/experiment8-9-10-comparison.pdf')

# %%
fig = sns.catplot(
    data=u.rename_for_display(ex8910_compare),
    y='sdr',
    x='condition',
    order=['Score: multi-source', 'Score: tenor', 'Score: SATB', 'No score: tenor', 'No score: SATB'],
    kind='strip', dodge=True,
#     kind='box', showfliers=False, width=0.4,
#     kind='box', fliersize=1,
#    kind='violin',
    aspect=1.4,
)
plt.xlabel(None);
#plt.xticks(rotation=90);
fig.savefig('figures/experiment8-9-10-comparison-strip.pdf')

# %% [markdown]
# #### Soprano

# %%
u.experiment_results(8, results).query('source == "soprano"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
u.experiment_results(10, results).query('source == "soprano"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
ex9_noscore_sop = results[results.model == '036'].fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 9 (no score)').assign(condition='No score: soprano')
ex8_best_sop = u.experiment_results(8, results).query('source == "soprano" & model == "082"').assign(condition='Score: SATB')
ex8_noscore_sop = results.query('model == "020" & source == "soprano"').fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 8 (no score)').assign(condition='No score: SATB')
#ex9_best = u.experiment_results(9, results).query('model == "071"').assign(condition='Score: tenor')
ex10_best_sop = u.experiment_results(10, results).query('model == "064"').assign(condition='Score: multi-source')

ex8910_compare_sop = pandas.concat([ex9_noscore_sop, ex8_noscore_sop, ex8_best_sop, ex10_best_sop])
fig = sns.catplot(
    data=u.rename_for_display(ex8910_compare_sop),
    y='sdr',
    x='condition',
    order=['Score: multi-source', 'Score: SATB', 'No score: soprano', 'No score: SATB'],
     kind='box', showfliers=False, width=0.4,
    aspect=1.4,
)
plt.xlabel(None);
#fig.savefig('figures/experiment8-9-10-comparison.pdf')

# %% [markdown]
# #### Alto

# %%
u.experiment_results(8, results).query('source == "alto"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
u.experiment_results(10, results).query('source == "alto"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
ex9_noscore_alto = results[results.model == '057'].fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 9 (no score)').assign(condition='No score: alto')
ex8_best_alto = u.experiment_results(8, results).query('source == "alto" & model == "084"').assign(condition='Score: SATB')
ex8_noscore_alto = results.query('model == "020" & source == "alto"').fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 8 (no score)').assign(condition='No score: SATB')
ex10_best_alto = u.experiment_results(10, results).query('model == "033"').assign(condition='Score: multi-source')

ex8910_compare_alto = pandas.concat([ex9_noscore_alto, ex8_noscore_alto, ex8_best_alto, ex10_best_alto])
fig = sns.catplot(
    data=u.rename_for_display(ex8910_compare_alto),
    y='sdr',
    x='condition',
    order=['Score: multi-source', 'Score: SATB', 'No score: alto', 'No score: SATB'],
     kind='box', showfliers=False, width=0.4,
    aspect=1.4,
)
plt.xlabel(None);
#fig.savefig('figures/experiment8-9-10-comparison.pdf')

# %% [markdown]
# #### Bass

# %%
u.experiment_results(8, results).query('source == "bass"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
u.experiment_results(10, results).query('source == "bass"').groupby(['model', 'score_type', 'score_concat']).sdr.median().sort_values(ascending=False).reset_index()

# %%
ex9_noscore_bass = results[results.model == '059'].fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 9 (no score)').assign(condition='No score: bass')
ex8_best_bass = u.experiment_results(8, results).query('source == "bass" & model == "083"').assign(condition='Score: SATB')
ex8_noscore_bass = results.query('model == "020" & source == "bass"').fillna({'score_type': 'no score', 'score_concat': 'no score'}).assign(experiment='Experiment 8 (no score)').assign(condition='No score: SATB')
ex10_best_bass = u.experiment_results(10, results).query('model == "022"').assign(condition='Score: multi-source')

ex8910_compare_bass = pandas.concat([ex9_noscore_bass, ex8_noscore_bass, ex8_best_bass, ex10_best_bass])
fig = sns.catplot(
    data=u.rename_for_display(ex8910_compare_bass),
    y='sdr',
    x='condition',
    order=['Score: multi-source', 'Score: SATB', 'No score: bass', 'No score: SATB'],
     kind='box', showfliers=False, width=0.4,
    aspect=1.4,
)
plt.xlabel(None);
#fig.savefig('figures/experiment8-9-10-comparison.pdf')
