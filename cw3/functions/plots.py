import matplotlib.pyplot as plt
import numpy as np

# plot multiple box plot, also add average values and labels
def myBoxplot(data, subxlabels, title="", xlabel="", ylabel=""):

    fig, ax = plt.subplots()
    bp = ax.boxplot(np.transpose(data))

    # Add a horizontal grid to the plot, but make it very light in color
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_xticklabels(subxlabels, rotation=45, fontsize=8)

    # plot the sample averages, with horizontal alignment
    # in the center of each box
    for i in range(len(data)):
        med = bp['medians'][i]
        ax.plot(np.average(med.get_xdata()), np.average(data[i]),
            color='w', marker='d', markeredgecolor='g')

    # get middle point of min and max value and plot range text
    for i in range(len(data)):
        med = bp['medians'][i]
        box_middle_x = np.average(med.get_xdata())
        max = np.max(data[i]) 
        min = np.min(data[i])
        shift = (max - min) / 100
        y = (max + min) / 2
        ax.plot(box_middle_x, y, color='w', marker='x', markeredgecolor='r')
        ax.text(box_middle_x, y - shift * 2, '%.4f\n± %.4f' % (y, (max - y)), 
            horizontalalignment='center', verticalalignment='top', fontsize=8)

    # legends
    fig.text(0.83, 0.97, '-', color='orange', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.973, ' Median', color='black', weight='roman',size='x-small')
    
    fig.text(0.83, 0.945, 'x', color='red', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.943, ' Min Max Center', color='black', weight='roman',size='x-small')
    
    fig.text(0.83, 0.915, '♢', color='green', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.913, ' Average', color='black', weight='roman',size='x-small')
    
    return fig, ax