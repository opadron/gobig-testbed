
import os

from argparse import ArgumentParser
from os import path

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

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
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--sizes", nargs="+", default=[], help="dataset sizes to process")

    args = parser.parse_args()
    args.sizes = set(args.sizes)

    return args

def main(args):
    path_prefix = "hdfs:///amplab/sequence"

    conf = None
    sc = None
    sql = None
    for size in args.sizes:
        timings = {}

        # uservisits
        # rankings
        # crawl

        visitors_table_name = "visitors_%s" % size
        rankings_table_name = "rankings_%s" % size
        crawl_table_name    = "crawl_%s"    % size

        if conf is None: conf = SparkConf()
        if sc is None: sc = SparkContext(conf=conf)
        if sql is None: sql = SQLContext(sc)

        tic()
        sql.createDataFrame(
            sc.sequenceFile(path.join(path_prefix, size, "uservisits"))
                .map(lambda x: tuple(x[1].split(",")))
                .map(lambda x: x[:4] + parse_agent(x[4]) + x[5:])
                .map(lambda x: Row(source_ip       =      (x[ 0]),
                                   url             =      (x[ 1]),
                                   date            = date (x[ 2]),
                                   revenue         = float(x[ 3]),
                                   os_name         =      (x[ 4]),
                                   os_version      =      (x[ 5]),
                                   browser_name    =      (x[ 6]),
                                   browser_version =      (x[ 7]),
                                   country         =      (x[ 8]),
                                   language        =      (x[ 9]),
                                   search          =      (x[10]),
                                   duration        = int  (x[11]),))
        ).registerTempTable(visitors_table_name)
        timings["open-and-register"] = toc()

        tic()
        os_results = sql.sql("""
            SELECT   os_name       AS os,
                     COUNT(FALSE)  AS total_visitors,
                     SUM(revenue)  AS total_revenue,
                     AVG(revenue)  AS average_revenue,
                     SUM(duration) AS total_duration,
                     AVG(duration) AS average_duration

            FROM     {}

            GROUP BY os_name
        """.format(visitors_table_name)).toPandas()
        timings["q-stats-by-os"] = toc()
        os_results.index = os_results.pop("os")

        tic()
        browser_results = sql.sql("""
            SELECT   browser_name  AS browser,
                     COUNT(FALSE)  AS total_visitors,
                     SUM(revenue)  AS total_revenue,
                     AVG(revenue)  AS average_revenue,
                     SUM(duration) AS total_duration,
                     AVG(duration) AS average_duration

            FROM     {}

            GROUP BY browser_name
        """.format(visitors_table_name)).toPandas()
        timings["q-stats-by-browser"] = toc()
        browser_results.index = browser_results.pop("browser")

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

