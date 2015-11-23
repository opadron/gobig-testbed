
from os import path
from sys import argv
from math import exp

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
import pandas as pd
import pandas as pd
import numpy as np
import mpmath as mp

mp.dps = 100

def make_bar_plot(data, title):
    data = data[sorted(data.columns)].sort_index().transpose()
    totals = np.array(data.sum(axis=1))
    data[:] = np.array(data)/totals[:, np.newaxis]

    plt.clf()
    data.plot(kind="barh", stacked=True)
    for index, total in enumerate(totals):
        plt.annotate(s=str(data.index[index]).replace("_", " "),
                     xy=(0.0, index + 0.35))
        plt.annotate(s="%g" % total, xy=(0.85, index + 0.35))

    plt.gca().axis("off")
    y0, y1 = plt.ylim()
    plt.xlim([0.0, 1.0])
    plt.ylim([y0, y1*1.1])
    plt.subplots_adjust(left=0.1, right=0.7, top=0.8, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.01, 0.895), loc="upper left")
    plt.title(title)

def main():
    base_directory = argv[1]
    data_size_keys = ["tiny", "1node", "5nodes"]
    data_size_labels = ["Small", "Medium", "Large"]

    for data_size, data_label in zip(data_size_keys, data_size_labels):
        for cluster_size in argv[2:]:
            for dataset_name in ("os", "browser"):
                for paradigm in ("spark", "mpi"):
                    make_bar_plot(
                        pd.read_pickle(path.join(base_directory,
                                                 data_size,
                                                 paradigm,
                                                 cluster_size,
                                                 dataset_name)),

                        "Visitor Stats by {} ({}-{})".format(
                            getattr(dataset_name, "upper"
                                    if len(dataset_name) <= 2
                                    else "capitalize")(),
                            paradigm,
                            cluster_size))

                    plt.gcf().savefig(
                        path.join(base_directory,
                                  data_size,
                                  paradigm,
                                  cluster_size,
                                  "{}.pdf".format(dataset_name)),
                        format="pdf")
                    plt.clf()
                    plt.close()

    timings_table = {}
    num_data_sizes = len(data_size_keys)

    for cluster_size in argv[2:]:
        data = np.empty((num_data_sizes, 2))
        for index, paradigm in enumerate(("spark", "mpi")):
            G = zip(range(num_data_sizes), data_size_keys, data_size_labels)
            for index2, data_size, data_label in G:
                timings_key = path.join(base_directory,
                                        data_size,
                                        paradigm,
                                        cluster_size,
                                        "timings")

                timings = timings_table.get(timings_key)
                if timings is None:
                    timings = dict(
                        (token[0].strip(), float(token[1].strip()))
                        for token in (
                            line.strip().split(",")
                            for line in open(timings_key))
                        if token)

                    timings_table[timings_key] = timings

                data[index2, index] = (timings["open-and-register"] +
                                       timings["q-stats-by-browser"] +
                                       timings["q-stats-by-os"])

        data_frame = pd.DataFrame(data=data,
                                  index=data_size_labels,
                                  columns=("spark", "mpi"))

        plt.figure(figsize=(13, 8))
        data_frame.plot(kind="bar")
        plt.ylabel("Run Time (s)")
        plt.xlabel("Dataset Size")
        plt.title(
            "Aggregate Statistics Scaling ({} Nodes)".format(cluster_size))

        plt.xticks(np.arange(num_data_sizes),
                   data_size_labels,
                   rotation="horizontal")

        plt.yscale("log", nonposy="clip")
        _, maxy = plt.ylim()
        plt.ylim(1e-1, 100*maxy)

        blue_patch = mpatches.Patch(edgecolor="black",
                                    facecolor="blue",
                                    label="spark")

        green_patch = mpatches.Patch(edgecolor="black",
                                     facecolor="green",
                                     label="mpi")

        xlim = plt.xlim()
        black_line = mlines.Line2D([-100, 100],
                                   [1, 1],
                                   color="black",
                                   linestyle="-")
        plt.xlim(xlim)
        plt.legend(handles=[blue_patch, green_patch])
        plt.gcf().savefig(path.join(base_directory,
                                    "scaling-{}.pdf".format(cluster_size)),
                          format="pdf")
        plt.clf()
        plt.close()

    for data_size, data_label in zip(data_size_keys, data_size_labels):
        data = np.empty((len(argv[2:]), 2))
        for index2, cluster_size in enumerate(argv[2:]):
            for index, paradigm in enumerate(("spark", "mpi")):
                timings_key = path.join(base_directory,
                                        data_size,
                                        paradigm,
                                        cluster_size,
                                        "timings")

                timings = timings_table.get(timings_key)
                if timings is None:
                    timings = dict(
                        (token[0].strip(), float(token[1].strip()))
                        for token in (
                            line.strip().split(",")
                            for line in open(timings_key))
                        if token)

                    timings_table[timings_key] = timings

                data[index2, index] = (timings["open-and-register"] +
                                       timings["q-stats-by-browser"] +
                                       timings["q-stats-by-os"])

            data_frame = pd.DataFrame(data=data,
                                      index=argv[2:],
                                      columns=("spark", "mpi"))

            plt.figure(figsize=(13, 8))
            data_frame.plot(kind="bar")
            plt.ylabel("Run Time (s)")
            plt.xlabel("Number of Nodes")
            plt.title(
                "Aggregate Statistics Scaling ({} Dataset)".format(data_label))

            plt.xticks(np.arange(len(argv[2:])),
                       argv[2:],
                       rotation="horizontal")

            plt.yscale("log", nonposy="clip")
            _, maxy = plt.ylim()
            plt.ylim(0.1, 100*maxy)

            blue_patch = mpatches.Patch(edgecolor="black",
                                        facecolor="blue",
                                        label="spark")

            green_patch = mpatches.Patch(edgecolor="black",
                                         facecolor="green",
                                         label="mpi")

            xlim = plt.xlim()
            black_line = mlines.Line2D([-100, 100],
                                       [1, 1],
                                       color="black",
                                       linestyle="-")
            plt.xlim(xlim)
            plt.legend(handles=[blue_patch, green_patch])
            plt.gcf().savefig(path.join(base_directory,
                                        "scaling-{}.pdf".format(
                                            data_label.lower())),
                              format="pdf")
            plt.clf()
            plt.close()

if __name__ == "__main__":
    from sys import exit
    exit(main())

