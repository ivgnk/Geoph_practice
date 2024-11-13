"""
Eventplot demo
Событийный график, показывающий последовательности событий с различными свойствами линий. График показан как в горизонтальной, так и в вертикальной ориентации.
https://matplotlib.org/stable/gallery/lines_bars_and_markers/eventplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-eventplot-demo-py
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def Eventplot_demo():
    matplotlib.rcParams['font.size'] = 8.0
    np.random.seed(125)
    ndat=4
    # create random data
    data1 = np.random.random([ndat, 50])
    # set different colors for each set of positions
    colors1 = [f'C{i}' for i in range(ndat)]
    # set different line properties for each set of positions
    # note that some overlap
    lineoffsets1 = [-15, -3, 1, 1.5]  # , 6, 10
    linelengths1 = [5, 2, 1, 1]  # , 3, 1.5
    list_of_names = ['n1', 'n2', 'n3', 'n4']
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.suptitle('Eventplot_demo')

    # create a horizontal plot
    axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                        linelengths=linelengths1)
    axs[0, 0].set_title('1');  axs[0, 0].grid()
    axs[0, 0].legend(list_of_names, loc='center right')

    # create a vertical plot
    axs[1, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                        linelengths=linelengths1, orientation='vertical')
    axs[1, 0].set_title('2');  axs[1, 0].grid()
    axs[1, 0].legend(list_of_names, loc='lower center')

    data2 = np.random.gamma(ndat, size=[60, 50])

    colors2 = 'green'; lineoffsets2 = 1; linelengths2 = 1

    # create a horizontal plot
    axs[0, 1].eventplot(data1, colors=colors2, lineoffsets=lineoffsets2,
                        linelengths=linelengths2)
    axs[0, 1].set_title('3');  axs[0, 1].grid()

    # create a vertical plot
    axs[1, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                        linelengths=linelengths2, orientation='vertical')
    axs[1, 1].set_title('4');  axs[1, 1].grid()
    plt.show()

if __name__=="__main__":
    Eventplot_demo()
