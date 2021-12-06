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

    for i in range(len(data)):
        med = bp['medians'][i]
        box_middle_x = np.average(med.get_xdata())
        
        max = np.max(data[i]) 
        min = np.min(data[i])
        vert_shift = (max - min) / 100
        max_all = np.max(data)
        min_all = np.min(data)
        
        # plot median text
        median = med.get_ydata()[i]
        ax.text(box_middle_x, min_all - vert_shift * 2, '%.4f\n' % median,
            horizontalalignment='center', verticalalignment='top', fontsize=8, color='orange')
                        
        # plot average
        mean = np.average(data[i])
        ax.plot(box_middle_x, mean,
            color='w', marker='d', markeredgecolor='g')
        # plot average text
        ax.text(box_middle_x, max_all + vert_shift * 3.5, '%.4f\n' % mean,
            horizontalalignment='center', verticalalignment='top', fontsize=8, color='green')
        
        # plot range
        y = (max + min) / 2
        ax.plot(box_middle_x, y, color='w', marker='x', markeredgecolor='r')
        # plot range text
        ax.text(box_middle_x, y - vert_shift * 2, '%.4f\n± %.4f' % (y, (max - y)), 
            horizontalalignment='center', verticalalignment='top', fontsize=8, color='red')

    # legends
    fig.text(0.83, 0.97, '-', color='orange', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.973, ' Median', color='black', weight='roman',size='x-small')
    
    fig.text(0.83, 0.945, 'x', color='red', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.943, ' Min Max Center', color='black', weight='roman',size='x-small')
    
    fig.text(0.83, 0.915, '♢', color='green', backgroundcolor='white', size='medium')
    fig.text(0.845, 0.913, ' Average', color='black', weight='roman',size='x-small')
    
    return fig, ax