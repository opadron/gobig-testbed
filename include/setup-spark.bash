
get_top_file() {
    ls -1 $1 | sort | tail -n 1
}

hadoop_conf_file="$( get_top_file "/opt/hadoop/*/libexec/hadoop-config.sh" )"
spark_conf_file="$( get_top_file "/opt/spark/*/conf/spark-env.sh" )"

source "$hadoop_conf_file"
source "$spark_conf_file"

if [ -z "$HADOOP_VERSION" ] ; then
    export HADOOP_VERSION="$(
        basename "$(
            dirname "$(
                dirname "$hadoop_conf_file" )" )" )"
fi

export PYTHONPATH="$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH"
export PYTHONPATH="$SPARK_HOME/python:$PYTHONPATH"

