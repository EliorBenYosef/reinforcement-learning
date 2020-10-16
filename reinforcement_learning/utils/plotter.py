import numpy as np
from matplotlib import pyplot as plt


# colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--', 'w--']

# colors = ['#FF0000', '#fa3c3c', '#E53729',
#           '#f08228', '#FB9946', '#FF7F00',
#           '#e6af2d',
#           '#e6dc32', '#FFFF00',
#           '#a0e632', '#00FF00',  '#00dc00',
#           '#17A858', '#00d28c',
#           '#00c8c8', '#0DB0DD',  '#00a0ff', '#1e3cff', '#0000FF',
#           '#6e00dc', '#8B00FF',  '#4B0082', '#a000c8', '#662371',
#           '#f00082']

colors = ['#FF0000', '#E53729',
          '#f08228', '#FF7F00',
          '#e6af2d',
          '#e6dc32', '#FFFF00',
          '#a0e632', '#00dc00',
          '#17A858', '#00d28c',
          '#00c8c8', '#1e3cff',
          '#6e00dc', '#a000c8',
          '#f00082']


def get_running_avg(scores, window):
    episodes = len(scores)

    if episodes >= window + 50:
        x = [i + 1 for i in range(window - 1, episodes)]

        running_avg = np.empty(episodes - window + 1)
        for t in range(window - 1, episodes):
            running_avg[t - window + 1] = np.mean(scores[(t - window + 1):(t + 1)])

    else:
        x = [i + 1 for i in range(episodes)]

        running_avg = np.empty(episodes)
        for t in range(episodes):
            running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

    return x, running_avg


def plot_running_average(env_name, method_name, scores, window=100, show=False, file_name=None, directory=''):
    plt.title(env_name + ' - ' + method_name + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(*get_running_avg(scores, window))
    if file_name:
        plt.savefig(directory + file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_accumulated_scores(env_name, method_name, scores, show=False, file_name=None, directory=''):
    plt.title(env_name + ' - ' + method_name + ' - Accumulated Score')
    plt.ylabel('Accumulated Score')
    plt.xlabel('Episode')
    x = [i + 1 for i in range(len(scores))]
    plt.plot(x, scores)
    if file_name:
        plt.savefig(directory + file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_running_average_comparison(main_title, scores_list, labels=None, window=100, show=False,
                                    file_name=None, directory=''):
    plt.figure(figsize=(8.5, 4.5))
    plt.title(main_title + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    # colors = []
    # for i in range(len(scores_list)):
    #     colors.append(np.random.rand(3, ))
    for i, scores in enumerate(scores_list):
        plt.plot(*get_running_avg(scores, window), colors[i])
    if labels:
        # plt.legend(labels)
        plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.subplots_adjust(right=0.7)
    if file_name:
        plt.savefig(directory + file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_accumulated_scores_comparison(main_title, scores_list, labels=None, show=False,
                                       file_name=None, directory=''):
    plt.figure(figsize=(8.5, 4.5))
    plt.title(main_title + ' - Accumulated Score')
    plt.ylabel('Accumulated Score')
    plt.xlabel('Episode')
    # colors = []
    # for i in range(len(scores_list)):
    #     colors.append(np.random.rand(3, ))
    for i, scores in enumerate(scores_list):
        x = [i + 1 for i in range(len(scores))]
        plt.plot(x, scores, colors[i])
    if labels:
        # plt.legend(labels)
        plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.subplots_adjust(right=0.7)
    if file_name:
        plt.savefig(directory + file_name + '.png')
    if show:
        plt.show()
    plt.close()
