import numpy as np
import matplotlib.pyplot as plt
import asyncio

plotdata = np.zeros((20000, 1))
plotdata_enhance = np.zeros((20000, 1))

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.set_ylim([-0.5, 0.5])
ax2.set_ylim([-0.5, 0.5])
lines1 = ax1.plot(plotdata)
lines2 = ax2.plot(plotdata_enhance)
lines = [lines1, lines2]


async def plot_trace(data, data_enhance):
    for column, line in enumerate(lines[0]):
        line.set_ydata(data[:])
    for column, line in enumerate(lines[1]):
        line.set_ydata(data_enhance[:])
    plt.savefig('templates/trace_plot.png')

loop = asyncio.get_event_loop()


