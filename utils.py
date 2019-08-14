import pandas
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import os
from IPython.display import display, Audio
import librosa
from ipywidgets import GridBox, Label, Layout, Output
import glob
import tabulate
import numpy as np
import soundfile

all_sources = ['soprano', 'alto', 'tenor', 'bass']
test_dir = 'evaluation'
sheet_id = os.environ['RESULTS_SHEET_ID']
datasets_dir = 'datasets'

source_dtype = pandas.CategoricalDtype(categories=['soprano', 'alto', 'tenor', 'bass'], ordered=True)

test_chorales = [
    '335', '336', '337', '338', '339', '340', '341', '342', '343', '345', '346',
    '349', '350', '351', '352', '354', '355', '356', '357', '358', '359', '360',
    '361', '363', '364', '365', '366', '367', '369', '370', '371'
]
dataset_sample_rates = {
    'chorales_synth_v5': 44100,
    'chorales_synth_v6': 22050,
}

experiment_models = {
    1: ['308440'],
    2: ['914434'],
    3: ['013', '007', '009', '011'],
    4: ['036', '057', '058', '059'],
    # 5: ['726622'] Not trained successfully
    6: ['012', '006', '008', '010'],
    7: ['016'],
    8: ['078', '079', '080', '081', '082', '083', '084', '085', '086', '090', '091', '092'],
    9: ['069', '070', '071', '072', '073', '074', '075', '076', '077', '087', '088', '089'],
    10: ['062', '064', '066', '026', '022', '033', '060', '068'],
}

def read_evaluation_results(test_dir=test_dir):
    model_evaluations = []
    for dataset_dir in Path(test_dir).iterdir():
        if not dataset_dir.is_dir():
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            evaluation_file = model_dir / 'evaluation.csv'
            if not evaluation_file.exists():
                continue
            evaluation = pandas.read_csv(evaluation_file, index_col=0, dtype={'model': str, 'checkpoint': str, 'source': source_dtype})
            evaluation.insert(2, 'evaluation_dataset', dataset_dir.name)
            model_evaluations.append(evaluation)
    
    return pandas.concat(model_evaluations, ignore_index=True)

def read_sheet(key, sheet_name, **kwargs):
    url = f'https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pandas.read_csv(url, **kwargs)

def read_models():
    models = read_sheet(sheet_id, 'Models', dtype={'model': str, 'filters_per_layer': 'Int64'}, skip_blank_lines=True)
    models = models[models.model.notnull()]
    models.set_index('model', inplace=True, verify_integrity=True)
    return models

def read_datasets():
    datasets = read_sheet(sheet_id, 'Datasets')
    datasets.set_index('dataset', inplace=True, verify_integrity=True)
    return datasets

def load_results():
    evaluation_results = read_evaluation_results()
    models = read_models()
    datasets = read_datasets()
    models = models.merge(datasets.add_prefix('training_dataset_'), left_on='dataset', right_index=True)
    results = evaluation_results.merge(models, left_on='model', right_index=True)
    results = results.merge(datasets.add_prefix('evaluation_dataset_'), left_on='evaluation_dataset', right_index=True)
    assert len(results) == len(evaluation_results), 'Some models or datasets are missing in the spreadsheet'
    id_vars = results.columns[~results.columns.isin(['sdr', 'isr', 'sir', 'sar'])]
    unpivoted = results.melt(id_vars=id_vars, var_name='metric', value_name='score')
    return results, unpivoted, models, datasets

source_abbrev_to_source_name = {
    'S': 'soprano',
    'A': 'alto',
    'T': 'tenor',
    'B': 'bass'
}

def source_abbrevs_to_source_names(abbrevs):
    return [source_abbrev_to_source_name[abbrev] for abbrev in abbrevs]

def load_audio_frame(filename, frame, context=1):
    output = Output(layout={'width': '100%', 'align-items': 'center', 'display': 'flex'})
    output.append_display_data(load_audio_frame_raw(filename, frame, context))
    return output

def load_audio_frame_raw(filename, frame, context):
    audio, sr = librosa.load(filename, sr=None, mono=False, offset=frame-context, duration=1+context)
    return Audio(audio, rate=sr, normalize=False)

def get_last_checkpoint(model_name, evaluation_dir):
    checkpoint_dirs = sorted(glob.glob(f'{evaluation_dir}/{model_name}-*'))
    last_checkpoint_dir = os.path.basename(checkpoint_dirs[-1])
    _model_name, checkpoint = last_checkpoint_dir.split('-', 1)
    return checkpoint

def get_estimate_filename(evaluation_dataset, model, chorale, source, multi_source, nmf=False):
    if nmf:
        return f'/Users/matan/gdrive/Grad/Thesis/Training/Wave-U-Net/nmf_evaluation/{evaluation_dataset}/{model}/{chorale}/{source}.wav'

    dataset_dir = f'/Users/matan/gdrive/Grad/Thesis/Training/Wave-U-Net/test/{evaluation_dataset}'
    checkpoint = get_last_checkpoint(model, dataset_dir)
    model_dir = f'{dataset_dir}/{model}-{checkpoint}'
    if multi_source:
        return f'{model_dir}/{chorale}/{source}/chorale_{chorale}_mix.wav_source.wav'
    else:
        return f'{model_dir}/{chorale}/chorale_{chorale}_mix.wav_{source}.wav'

def display_frame(evaluation_dataset, sources, chorale, frame, model, models, context=1, context_separate=False, nmf=False):
    dataset_dir = f'/Users/matan/gdrive/Grad/Thesis/Data/Synthesized/{evaluation_dataset}/'
    mix = f'{dataset_dir}/mix/chorale_{chorale}_mix.wav'
    items = [Label(l) for l in ['Name', 'Reference', 'Estimate']]
    items += [Label('Mix (context)'), load_audio_frame(mix, frame, context), Label('')]
    if context_separate:
        items += [Label('Mix'), load_audio_frame(mix, frame, 0), Label('')]
    for source in sources:
        reference = f'{dataset_dir}/audio_mono/chorale_{chorale}_{source}.wav'
        multi_source = False if nmf else models.loc[model].multi_source
        estimate = get_estimate_filename(evaluation_dataset, model, chorale, source, multi_source, nmf)
        items += [Label(source + ' (context)'), load_audio_frame(reference, frame, context), load_audio_frame(estimate, frame, context)]
        if context_separate:
            items += [Label(source), load_audio_frame(reference, frame, 0), load_audio_frame(estimate, frame, 0)]
    display(GridBox(items, layout={'grid_template_columns': '120px 350px 350px', 'justify-items': 'center'}))

def markdown_table(df, **kwargs):
    return tabulate.tabulate(df, headers="keys", floatfmt=".2f", **kwargs)

def get_snr(reference_frame, estimated_frame, regularization=0):
    signal_energy = np.sum(reference_frame ** 2, axis=0) + regularization
    noise_energy = np.sum((reference_frame - estimated_frame) ** 2, axis=0)
    return librosa.power_to_db(signal_energy / noise_energy)

def load_frame(evaluation_dataset, source, chorale, frame, model, models):
    dataset_dir = f'{datasets_dir}/{evaluation_dataset}'
    reference = f'{dataset_dir}/audio_mono/chorale_{chorale}_{source}.wav'
    estimate = get_estimate_filename(evaluation_dataset, model, chorale, source, models.loc[model].multi_source)
    return read_frame(reference, frame, evaluation_dataset), read_frame(estimate, frame, evaluation_dataset)

def get_frame_snr(evaluation_dataset, source, chorale, frame, model, models):
    dataset_dir = f'/Users/matan/gdrive/Grad/Thesis/Data/Synthesized/{evaluation_dataset}/'
    mix = f'{dataset_dir}/mix/chorale_{chorale}_mix.wav'

def display_reference_frame(dataset, source, chorale, frame, context=0):
    dataset_dir = f'/Users/matan/gdrive/Grad/Thesis/Data/Synthesized/{dataset}/'
    reference = f'{dataset_dir}/audio_mono/chorale_{chorale}_{source}.wav'
    return load_audio_frame_raw(reference, frame, context)

def read_chorale(dataset, chorale, source):
    filename = f'{datasets_dir}/{dataset}/audio_mono/chorale_{chorale}_{source}.wav'
    audio, sr = soundfile.read(filename)
    return audio, sr

def get_frame_powers():
    frame_powers_dfs = []
    for dataset in ['chorales_synth_v5', 'chorales_synth_v6']:
        print(dataset)
        dataset_sr = dataset_sample_rates[dataset]
        # One second frames
        frame_size = dataset_sr
        for chorale in test_chorales:
            print(chorale)
            for source in all_sources:
                filename = f'{datasets_dir}/{dataset}/audio_mono/chorale_{chorale}_{source}.wav'
                audio, audio_sr = soundfile.read(filename)
                assert audio_sr == dataset_sr, f'Expected sample rate {dataset_sr} but got {audio_sr} in file {filename}'
                frames = librosa.util.frame(audio, frame_size, frame_size)
                frame_powers = (frames ** 2).sum(axis=0) / frame_size
                frame_powers_df = pandas.DataFrame(frame_powers, columns=['power']).assign(dataset=dataset, chorale=int(chorale), source=source)
                frame_powers_df['frame'] = frame_powers_df.index
                frame_powers_dfs.append(frame_powers_df)
                
    return pandas.concat(frame_powers_dfs, ignore_index=True).astype({'source': source_dtype})

def get_frame_energies():
    frame_energies_dfs = []
    dataset_sr = frame_size = 22050
    for dataset in ['chorales_synth_v5', 'chorales_synth_v6']:
        print(dataset)
        suffix = '_22050Hz' if dataset == 'chorales_synth_v5' else ''
        # One second frames
        for chorale in test_chorales:
            print(chorale)
            for source in all_sources:
                filename = f'{datasets_dir}/{dataset}/audio_mono{suffix}/chorale_{chorale}_{source}.wav'
                audio, audio_sr = soundfile.read(filename)
                assert audio_sr == dataset_sr, f'Expected sample rate {dataset_sr} but got {audio_sr} in file {filename}'
                frames = librosa.util.frame(audio, frame_size, frame_size)
                frame_energies = (frames ** 2).sum(axis=0)
                frame_energies_df = pandas.DataFrame(frame_energies, columns=['energy']).assign(dataset=dataset, chorale=int(chorale), source=source)
                frame_energies_df['frame'] = frame_energies_df.index
                frame_energies_dfs.append(frame_energies_df)
                
    return pandas.concat(frame_energies_dfs, ignore_index=True).astype({'source': source_dtype})

def get_max_abs():
    max_abs = 0
    for dataset in ['chorales_synth_v5', 'chorales_synth_v6']:
        print(dataset)
        dataset_sr = dataset_sample_rates[dataset]
        for chorale in test_chorales:
            print(chorale)
            for source in all_sources:
                filename = f'{datasets_dir}/{dataset}/audio_mono/chorale_{chorale}_{source}.wav'
                audio, audio_sr = soundfile.read(filename)
                assert audio_sr == dataset_sr, f'Expected sample rate {dataset_sr} but got {audio_sr} in file {filename}'
                source_max_abs = np.abs(audio).max()
                if source_max_abs > max_abs:
                    max_abs = source_max_abs

    return max_abs

def read_frame(filename, frame, dataset, context=0):
    dataset_sr = dataset_sample_rates[dataset]
    audio, sr = librosa.load(filename, sr=None, mono=False, offset=frame-context, duration=1+context)
    assert sr == dataset_sr, f'Expected sample rate {dataset_sr} but got {sr} in file {filename}'
    return audio

def display_frame_from_row(row, models, context=1, context_separate=False, details=True, nmf=False):
    if details:
        display(row[['chorale', 'frame', 'source', 'model', 'sdr']].to_frame().T)
    display_frame(row.evaluation_dataset, [row.source], row.chorale, row.frame, row.model, models, context, context_separate, nmf)

def evaluate_snr(reference_sources, estimated_sources, frame_size_samples, regularization=0):
    num_sources, _num_samples = reference_sources.shape
    source_snrs = []
    for i in range(num_sources):
        reference_frames = librosa.util.frame(reference_sources[i, :], frame_size_samples, frame_size_samples)
        estimate_frames = librosa.util.frame(estimated_sources[i, :], frame_size_samples, frame_size_samples)
        signal_power = np.sum(reference_frames ** 2, axis=0) + regularization
        noise_power = np.sum((reference_frames - estimate_frames) ** 2, axis=0)
        snr = librosa.power_to_db(signal_power / noise_power)
        source_snrs.append(snr)
    return np.stack(source_snrs)

def evaluate_sdr_chorale(evaluation_dataset, model, chorale, source, models, regularization=0):
    import museval
    dataset_sr = dataset_sample_rates[evaluation_dataset]
    frame_size = dataset_sr
    estimate_filename = get_estimate_filename(evaluation_dataset, model, chorale, source, models.loc[model].multi_source)
    estimate, estimate_sr = soundfile.read(estimate_filename)
    reference, reference_sr = read_chorale(evaluation_dataset, chorale, source)
    assert reference_sr == estimate_sr == dataset_sr
    assert reference.shape == estimate.shape
    model_params = models.loc[model]
    reference_sources = np.array([reference])
    estimated_sources = np.array([estimate])
    sdr, _isr, _sir, _sar = museval.evaluate(reference_sources, estimated_sources, padding=False, win=frame_size, hop=frame_size)
    snr = evaluate_snr(reference_sources, estimated_sources, frame_size)
    snr_reg = evaluate_snr(reference_sources, estimated_sources, frame_size, regularization)
    assert sdr.shape == snr.shape == snr_reg.shape
    return pandas.DataFrame({'sdr_bsseval': sdr[0], 'sdr_mine': snr[0], 'sdr_reg': snr_reg[0], 'frame': range(sdr.shape[1])})


def evaluate_sdr_models(evaluation_dataset, models_to_evaluate, models, regularization):
    metrics: list = []
    for model in models_to_evaluate:
        print(f'Model {model}')
        for chorale in test_chorales:
            print(f'\tChorale {chorale}')
            for source in source_abbrevs_to_source_names(models.loc[model].extracted_sources):
                results = evaluate_sdr_chorale(evaluation_dataset, model, chorale, source, models, regularization)
                results_extra = results.assign(evaluation_dataset=evaluation_dataset, chorale=int(chorale), model=model, source=source)
                metrics.append(results_extra)

    return pandas.concat(metrics, ignore_index=True)

def experiment_results(experiment, results):
    res = results[results.model.isin(experiment_models[experiment])]
    return res.assign(experiment=f'Experiment {experiment}')

def experiments_results(experiments, results):
    return pandas.concat([
        experiment_results(experiment, results)
        for experiment in experiments
    ])

def rename_for_display(results):
    return results.replace({
        'score_concat': {
            'in': 'input',
            'out': 'output',
            'in-out': 'input-output'
        },
        'score_type': {
            'one-hot': 'piano roll',
            'pure tone synth': 'pure tone',
            'midi norm': 'normalized pitch'
        }
    }).rename(columns={
        'score_type': 'Score Type',
        'score_concat': 'Conditioning Location',
        'sdr': 'SDR (dB)'
    })