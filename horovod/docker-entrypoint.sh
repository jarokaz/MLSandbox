#!/bin/bash

# Check if running on a cluster
if [ -z "$CLUSTER_SPEC"]
  echo "Running on a single node"
  exit 0

ROLE=$(echo $CLUSTER_SPEC | jq '."task"."type"')

echo $ROLE

if [ "$ROLE" = ""master"" ]
then 
  echo "master"
else
  echo "worker"
fi






