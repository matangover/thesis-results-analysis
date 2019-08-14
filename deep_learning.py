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
import numpy, matplotlib.pyplot as plt
import seaborn

# %%
seaborn.set(font_scale=1.3)
seaborn.set_style('whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax in axes:
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_ylim(-1,1)
    ax.set_xlim(-3,3)
    ax.set_yticks(numpy.linspace(-2,2,5))

x = numpy.linspace(-3,3,1000)

sigmoid =  1 / (1+numpy.e**(-x))
axes[0].plot(x, sigmoid, linewidth=3)
axes[0].set_title('logistic sigmoid', pad=15)

tanh =  numpy.tanh(x)
axes[1].plot(x, tanh, linewidth=3)
axes[1].set_title('hyperbolic tangent', pad=15)

relu =  numpy.maximum(x, 0)
axes[2].plot(x, relu, linewidth=3)
axes[2].set_title('ReLU', pad=15)

plt.savefig('activation_functions.pdf', bbox_inches='tight')
