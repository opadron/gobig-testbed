
source "$( ls -1 /opt/spark/*/conf/spark-env.sh | sort | tail -n 1 )"

export PYTHONPATH="$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH"
export PYTHONPATH="$SPARK_HOME/python:$PYTHONPATH"

