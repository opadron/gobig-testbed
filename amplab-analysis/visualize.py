
from os import path
from sys import argv

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def make_plot(data, title):
    data = data.transpose()
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
    for directory in argv[1:]:
        for dataset_name in ("os", "browser"):
            make_plot(
                pd.read_pickle(path.join(directory, dataset_name)),
                "Visitor Stats by {}".format(
                    getattr(dataset_name, "upper"
                            if len(dataset_name) <= 2
                            else "capitalize")()))
            plt.gcf().savefig(
                path.join(directory, "{}.pdf".format(dataset_name)),
                format="pdf")

if __name__ == "__main__":
    from sys import exit
    exit(main())

