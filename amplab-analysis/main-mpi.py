
from mpi4py import MPI

import os
import itertools as it

from argparse import ArgumentParser
from os import path
from xml.dom import minidom

import numpy as np
import pandas as pd

from snakebite.client import Client as HDFS

c_rank = MPI.COMM_WORLD.Get_rank()
c_size = MPI.COMM_WORLD.Get_size()

class toc(object):
    from time import time
    def __init__(self): self.stamp = None
    def tic(self): self.stamp = self.__class__.time()
    def toc(self): return self.__class__.time() - self.stamp
toc = toc() ; tic = toc.tic ; toc = toc.toc

def parse_agent(agent):
    from httpagentparser import detect
    result = detect(agent)

    os = result.get("os", {})
    os_name = os.get("name", "?")
    os_version = os.get("version", "?")

    browser = result.get("browser", {})
    browser_name = browser.get("name", "?")
    browser_version = browser.get("version", "?")

    while browser_name.lower().startswith("microsoft"):
        browser_name = " ".join(browser_name.split()[1:])

    while browser_name.lower().startswith("internet explorer"):
        browser_name = " ".join(browser_name.split()[1:])

    if os_name.lower() == "linux":
        dist = result.get("dist", {})
        os_name = dist.get("name", os_name)
        os_version = dist.get("version", os_version)

    return os_name, os_version, browser_name, browser_version

def date(date):
    from datetime import datetime
    return datetime.strptime(date, "%Y-%m-%d")

def mkdir_p(path):
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def parse_args():
    args = None
    if c_rank == 0:
        parser = ArgumentParser()
        parser.add_argument(
            "-s", "--sizes", nargs="+", default=[],
            help="dataset sizes to process")

        try:
            args = parser.parse_args()
        except:
            MPI.COMM_WORLD.bcast(None, root=0)
            raise
        args.sizes = set(args.sizes)

    args = MPI.COMM_WORLD.bcast(args, root=0)
    return args

def row_iterator(file_list, hdfs):
    result = (
        ((column[0]), (column[1]), date(column[ 2]), float(column[ 3]),
         (column[4]), (column[5]),     (column[ 6]),      (column[ 7]),
         (column[8]), (column[9]),     (column[10]), int  (column[11]))
        for column in (
            (columns[:4] + parse_agent(columns[4]) + columns[5:])
            for columns in (
                tuple(line.split(","))
                for line in
                    it.chain.from_iterable(text.split("\n")
                                           for text in hdfs.text(file_list))
                if line.strip()
            )
        )
    )

    return result

def reduce_operator(a, b):
    result = {}
    for key in set(a.keys()) | set(b.keys()):
        a_result = a.get(key, [0, 0, 0])
        b_result = b.get(key, [0, 0, 0])
        result[key] = [x + y for x, y in zip(a_result, b_result)]
    return result

def reduce_data(table, group_by_index, group_by_name):
    partial_result = {}
    for row in table:
        _, _, _, revenue, _, _, _, _, _, _, _, duration = row
        group_by_value = row[group_by_index]

        group_result = partial_result.get(group_by_value)
        if group_result is None:
            group_result = [0, 0, 0]
            partial_result[group_by_value] = group_result

        group_result[0] += 1
        group_result[1] += revenue
        group_result[2] += duration

    global_result = MPI.COMM_WORLD.reduce(partial_result,
                                          op=reduce_operator,
                                          root=0)

    if c_rank == 0:
        group_by_array = np.array(list(global_result.keys()))

        total_visitors   = np.empty_like(group_by_array, dtype=np.int64)
        total_revenue    = np.empty_like(group_by_array, dtype=np.float64)
        average_revenue  = np.empty_like(group_by_array, dtype=np.float64)
        total_duration   = np.empty_like(group_by_array, dtype=np.float64)
        average_duration = np.empty_like(group_by_array, dtype=np.float64)

        for i, key in enumerate(group_by_array):
            count, revenue, duration = global_result[key]
            total_visitors[i]   = count
            total_revenue[i]    = revenue
            total_duration[i]   = duration
            average_revenue[i]  = total_revenue[i]/count
            average_duration[i] = total_duration[i]/count

        global_result = pd.DataFrame({
            group_by_name     : group_by_array,
            "total_visitors"  : total_visitors,
            "total_revenue"   : total_revenue,
            "average_revenue" : average_revenue,
            "total_duration"  : total_duration,
            "average_duration": average_duration})

        return global_result

    return None

def main(args):
    xml = minidom.parse(path.join(os.environ["HADOOP_HOME"],
                                  "etc", "hadoop", "hdfs-site.xml"))

    element = [ x for x in xml.getElementsByTagName("property")
                if (x.getElementsByTagName("name")[0]
                     .childNodes[0]
                     .nodeValue == "dfs.namenode.http-address") ][0]

    namenode = (element.getElementsByTagName("value")[0]
                       .childNodes[0]
                       .nodeValue.split(":")[0])

    fs = HDFS(namenode, 8020)

    path_prefix = "/amplab/text"
    for size in args.sizes:
        timings = {}

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: tic()

        file_list = None
        if c_rank == 0:
            file_list = [
                entry["path"] for entry in fs.ls([
                    path.join(path_prefix, size, "uservisits")])]
            file_list = [file_list[i::c_size] for i in range(c_size)]

        file_list = MPI.COMM_WORLD.scatter(file_list, root=0)

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: timings["open-and-register"] = toc()

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: tic()

        os_results = reduce_data(row_iterator(file_list, fs), 4, "os")

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: timings["q-stats-by-os"] = toc()
        if c_rank == 0: os_results.index = os_results.pop("os")

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: tic()

        browser_results = reduce_data(row_iterator(file_list, fs), 6, "browser")

        MPI.COMM_WORLD.Barrier()
        if c_rank == 0: timings["q-stats-by-browser"] = toc()
        if c_rank == 0: browser_results.index = browser_results.pop("browser")

        if c_rank == 0:
            mkdir_p(path.join("results", size))
            with open(path.join("results", size, "timings"), "w") as f:
                for entry in timings.items():
                    f.write("%s, %.18e\n" % entry)
                f.flush()

            browser_results.to_pickle(path.join("results", size, "browser"))
            os_results.to_pickle(path.join("results", size, "os"))

    return 0

if __name__ == "__main__":
    from sys import exit
    exit(main(parse_args()))

