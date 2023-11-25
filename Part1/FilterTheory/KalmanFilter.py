# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Kalman Filter
#
# Reference
# - [An Introduction to the Kalman Filter](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
# - [Derivation of the Kalman filter in a Bayesian filtering perspective](https://ieeexplore.ieee.org/document/9581918)

# %% [markdown]
# Kalman Filter is a kind of time series estimation tasks.
# This theory tries to estimate hidden states from observable states.
#
#
# ### Formulation
# State Equation
# \begin{equation}
# x_t = A x_{t-1} + w_{t-1}
# \end{equation}
#
# Measurement Equation
# \begin{equation}
# y_t = Hx_t + e_t
# \end{equation}
#
# with $w_t \sim N(0, Q), e_t \sim N(0, R)$
#
# Thanks of linearity, $x_t$ also obeys gaussian distribution.
# The goal is to figure out the posterior distribution of $x_t$ before and after observing $y_t$.
#
# ### Main Result
#
# Kalman Filter involves 2 blocks mainly, time update equation("Predict") and measurement update equation("Correct").
#
# #### time update
#
# \begin{equation}
# x(t|t-1) = Ax(t-1|t-1)
# \end{equation}
#
# \begin{equation}
# S(t|t-1) = AS(t-1|t-1)A^T + Q
# \end{equation}
#
# #### measurement update
#
# \begin{equation}
# x(t|t) = x(t|t-1) + K_t(y_t - Hx_t)
# \end{equation}
#
# \begin{equation}
# S(t|t) = (I - K_tH)S(t|t-1)
# \end{equation}
#
# \begin{equation}
# K_t = S(t|t-1)H^T(HS(t|t-1)H^T + R)^{-1}
# \end{equation}

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Ex1) Kalman Filter with 1 dimension

# %%
# System Parameter
a = 1.0
q = 1.0
h = 1.0
r = 0.5
x0 = 1.0

T = 100
times = np.arange(T+1)


# %%
def t_update_1dim(x, s, a, noise):
    next_x = a * x
    next_s = a * s * a + noise
    return next_x, next_s

def m_update_1dim(x, s, y, h, r):
    noise = np.random.normal(loc=0.0, scale=r)
    gain = s * h / (h * s * h + noise)
    modified_x = x + gain * (y - h * x)
    modified_s = (1 - gain * h) * s
    return modified_x, modified_s


# %%
pred_init = 0.5
vol_init = 1.0

xs = [x0]
preds = [pred_init]
vols = [vol_init]
for t in range(1, T+1):
    # sysytem update
    noise = np.random.normal(loc=0.0, scale=q)
    x = a * xs[-1] + noise
    xs.append(x)
    y = h * x
    
    # time update
    pred, vol = t_update_1dim(preds[-1], vols[-1], a, noise)
    # measurement update
    pred, vol = m_update_1dim(pred, vol, y, h, r)
    preds.append(pred)
    vols.append(vol)

# %%
plt.figure(figsize=(16, 6))
plt.plot(times, xs, color='r', label="True States")
plt.plot(times, preds, color='b', label="Pred States")
plt.legend()

# %%

# %%
