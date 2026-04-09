# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:22:28 2026

@author: wyatt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


input_name = "BaselineElevatedView"
df = pd.read_csv(f"data/{input_name}_tracking.csv")

df = df[df['real_y'] <= 60]  # NET_Y or half-court length

x = df['adj_x']  # x relative to court center
y = df['real_y']  # y in feet

nbins = 300

k = gaussian_kde([x, y])

xi, yi = np.mgrid[
    -30:30:nbins*1j,
    -21:39:nbins*1j
]

zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

norm_zi = zi/zi.sum() # To compare with other 

zi = (zi - zi.min() / (zi.max() - zi.min())) # for the visuals


fig, ax = plt.subplots(figsize=(8, 12))

# Heatmap
ax.pcolormesh(xi, yi, zi, shading='auto')

# Match court orientation
ax.set_xlim(-30, 30)
ax.set_ylim(-21, 39)

service_line = 21
net = 39

# Doubles court
ax.add_patch(plt.Rectangle(
    (-18, 0),   # bottom-left corner
    36,         # width
    39,         # half court length
    fill=False,
    color='white',
    linewidth=2
))

# Singles court
ax.add_patch(plt.Rectangle(
    (-13.5, 0),   # bottom-left corner
    27,         # width
    39,         # half court length
    fill=False,
    color='white',
    linewidth=2
))

# Net
ax.axhline(net, color='white', linewidth=2)

# Service line
ax.add_patch(plt.Rectangle(
    (-13.5, 21),   # bottom-left corner
    27,         # width
    39,         # half court length
    fill=False,
    color='white',
    linewidth=2
))

# Center service line
ax.plot([0, 0], [service_line, net], color='white', linewidth=1)

ax.tick_params(axis='both', labelsize=16)

ax.set_xlabel('Distance From Center (ft)', fontsize = 30)
ax.set_ylabel('Distance From Baseline (ft)', fontsize = 30)
ax.set_title("Heatmap of Player's Court Position", fontsize = 30)


plt.show()