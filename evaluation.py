import argparse
import matplotlib.pyplot as plt
import json
import os
import pathlib
import fnmatch


def plot_monitor_output(file, title):
    with open(file, 'r') as f:
        results = json.load(f)
    fig, ax = plt.subplots()

    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.set_xlabel('Episode')

    plt.plot(results['episode_rewards'])
    plt_out = file.parent / 'reward_plot.png'
    plt.savefig(plt_out, dpi=600)
    print("Plot saved as: " + str(plt_out))
    plt.show()


def get_monitor_filename(directory):
    suffix_re = '*.stats.json'
    filename = ''
    for _, _, f in os.walk(directory):
        for name in f:
            if fnmatch.fnmatch(name, suffix_re):
                filename = directory / name
                break

    assert filename

    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plots the evaluation of a Lunar Lander model from the gym monitor output')
    parser.add_argument('-l', '--load',
                        help='Directory containing the gym monitor output', required=True)
    parser.add_argument('-t', '--title',
                        help='Title for the plot', required=True)
    args = parser.parse_args()

    filepath = get_monitor_filename(pathlib.Path(__file__).parent.absolute() / args.load)

    plot_monitor_output(filepath, args.title)
