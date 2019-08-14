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
# %matplotlib inline
import numpy, scipy, matplotlib.pyplot as plt, sklearn, IPython.display as ipd
import librosa, librosa.display
import ipywidgets
import mido
from typing import Iterable

# %%
# %load_ext autoreload
# %autoreload 1
# %aimport nmf
# %aimport separate_chorales

import nmf
import separate_chorales
import itertools

# %%
def separate_with_config(config_name):
    configs_by_name = {c.config_name: c for c in separate_chorales.configs}
    config = configs_by_name[config_name]
    audio = nmf.load_audio('audio/BWV_255_no_effect.flac', n_fft=config.n_fft, display=False)
    score = mido.MidiFile('audio/BWV_255.mid')
    W_init = nmf.initialize_components_midi(audio, score, phi=config.phi, num_partials=config.num_partials)
    H_init = nmf.initialize_activations_midi(audio, score, tol_on=config.tol_on, tol_off=config.tol_off)
    nmf.score_informed_nmf_display_midi(audio, score, W_init, H_init, tol_on=config.tol_on, tol_off=config.tol_off, spectrogram_ylim=10000)


# %%
separate_with_config('A')

# %%
separate_with_config('B')

# %%
separate_with_config('C')

# %%
separate_with_config('D')

# %% [markdown]
# ### Check if our window is valid for resynthesis
#
# Yes. Hann window with 75% overlapping is fine.

# %%
scipy.signal.check_COLA('hann', 2048, 2048*0.75)


# %% [markdown]
# ### NMF explanation figure

# %%
def initialize_components(signal, pitches, num_partials=15, phi=0.5):
    n_features, _n_samples = signal.S.shape
    fft_freqs = librosa.fft_frequencies(signal.sr, signal.n_fft)
    n_components = len(pitches)
    W_init = numpy.zeros((n_features, n_components))
    phi_below = phi if isinstance(phi, (float, int)) else phi[0]
    phi_above = phi if isinstance(phi, (float, int)) else phi[1]
    for i, pitch in enumerate(pitches):
        if num_partials is None:
            partials: Iterable[int] = itertools.count(start=1)
        else:
            partials = range(1, num_partials + 1)
        for partial in partials:
            min_freq = librosa.midi_to_hz(pitch - phi_below) * partial
            if min_freq > fft_freqs[-1]:
                break
            max_freq = librosa.midi_to_hz(pitch + phi_above)  * partial
            max_freq = min(fft_freqs[-1], max_freq)
            intensity = 4 / (partial+3)
            start_bin = nmf.freq_to_bin(min_freq, fft_freqs, round='down')
            end_bin = nmf.freq_to_bin(max_freq, fft_freqs, round='up')
            W_init[start_bin:end_bin+1,i] = intensity
    return W_init


# %%
n_fft = 1024
n_frames = int(5*22050 / n_fft)
pitches = [54, 63, 71]
signal = nmf.Signal(None, 22050, numpy.zeros((n_fft, n_frames)), None, None, n_fft, n_fft)
components = initialize_components(signal, pitches, 200, 0)


n_components = len(pitches)
_n_features, n_samples = signal.S.shape
h = numpy.zeros((n_components, n_samples))
h[0,5:12] = 1
h[2,15:30] =1
h[1,35:45] =1
h[0,52:72] = numpy.linspace(0.8, 0.2, 20)
h[1,60:70] = numpy.linspace(1, 0.2, 10)
h[2,75:90] = numpy.linspace(1, 0, 15)
h[0,80:100] = numpy.linspace(1, 0.5, 20)
# %%
def plot_components():
    librosa.display.specshow(components, sr=signal.sr, y_axis='linear')
    plt.ylim((0, 1000))
    plt.xticks([0.5,1.5,2.5], ['A', 'B', 'C'])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Component')
    plt.savefig('components.pdf', bbox_inches='tight')

# %%
def plot_activations():
    cmap = librosa.display.cmap(h)
    librosa.display.specshow(h, sr=signal.sr, x_axis='time', hop_length=signal.fft_hop_length, cmap=cmap)
    plt.yticks([0.5, 1.5, 2.5], ['A', 'B', 'C'])
    plt.xlabel(None)
    plt.ylabel('Component')
    plt.gca().invert_yaxis()
    plt.savefig('activations.pdf', bbox_inches='tight')


# %%
def plot_spec():
    librosa.display.specshow(numpy.dot(components, h), sr=signal.sr, y_axis='linear', hop_length=signal.fft_hop_length, x_axis='time')
    plt.ylim((0, 1000))
    plt.ylabel(None)
    plt.yticks([])
    plt.xlabel('Time (seconds)')
    plt.savefig('spec.pdf', bbox_inches='tight')


# %%
fig, axes = plt.subplots(2, 2, sharex='col', sharey=False, figsize=(14,8),
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.03,
                                    'width_ratios': [1, 5],
                                    'height_ratios': [1, 5]})
fig.delaxes(axes[0, 0])
plt.sca(axes[0, 1])
plot_activations()
plt.sca(axes[1, 0])
plot_components()
plt.sca(axes[1, 1])
plot_spec()

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.22, 0.015, 0.4])
mappable = axes[1,1].collections[0]
fig.colorbar(mappable,cax=cbar_ax)
plt.savefig('nmf.pdf', bbox_inches='tight')
