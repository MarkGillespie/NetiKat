#!/usr/local/bin/python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import math
import sys
import operator
import numpy as np


sparseVecData = np.genfromtxt('sparse_vector_data.tsv', delimiter='\t', dtype=[('size', 'i8'), ('skip', 'f8'), ('drop', 'f8'), ('test', 'f8'), ('testSize', 'f8'), ('set', 'f8'), ('amp', 'f8'), ('seq', 'f8'), ('choice', 'f8'), ('star', 'f8')])
sparseMatData = np.genfromtxt('sparse_matrix_data.tsv', delimiter='\t', dtype=[('size', 'i8'), ('skip', 'f8'), ('drop', 'f8'), ('test', 'f8'), ('testSize', 'f8'), ('set', 'f8'), ('amp', 'f8'), ('seq', 'f8'), ('choice', 'f8'), ('star', 'f8')])

fig = plt.figure()
ax1 = fig.add_subplot(111)

operations = ['seq', 'amp', 'star', 'testSize', 'skip']

# https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452#8391452
def setPlotColors(mapName):
    NUM_COLORS = len(operations)

    cmap = plt.get_cmap(mapName)
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS+2)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    colorSeq = [scalarMap.to_rgba(i) for i in range(NUM_COLORS)]
    markerSeq = ['o', '^', 's', 'd', 'P']
    ax1.set_prop_cycle(color=colorSeq, marker=markerSeq)

setPlotColors('viridis')
for op in operations:
    ax1.plot(sparseVecData['size'], sparseVecData[op], label=op)

setPlotColors('autumn')
for op in operations:
    ax1.plot(sparseMatData['size'], sparseMatData[op], label=op)

ax1.grid(True)

# Shrink figure to put legend outside plot
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Draw two separate legends
# https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
lines = ax1.get_lines()
legend1 = ax1.legend(lines[0:5], operations, loc='upper left', title='Sparse Vector Operations', bbox_to_anchor=(1, 1))
legend2 = ax1.legend(lines[5:10], operations, loc='lower left', title='Sparse Matrix Operations', bbox_to_anchor=(1, 0))
ax1.add_artist(legend1)
ax1.add_artist(legend2)

ax1.set_title('NetiKAT Performance')
ax1.set_xlabel('Matrix dimension')
ax1.set_ylabel('Time (s)')

# plt.xscale('log')
# plt.yscale('log')
plt.show()

