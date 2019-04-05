#!/bin/bash

#!/bin/bash

# Get or set command line parameters
NP="${1:-1}"
SCRIPT="${2:-train.py}"

# Check if running on CMLE
if [ -z "$CLUSTER_SPEC" ]
then
   echo "Not running on CMLE"
   exit 0
fi

# Extract the role   
ROLE=$(echo $CLUSTER_SPEC | jq '."task"."type"')

echo $ROLE

# Use CMLE master as a horovod's primary worker and CMLE workers
# as horovod's secondary workers
if [ "$ROLE" = '"master"' ]
then 
  
 # Preprocess CLUSTER_SPEC 
 # JQUERY='."cluster"."master" + ."cluster"."worker" |
 #         {hosts: map(sub(":[0-9]{4}"; "")), 
 #          port: (.[0] | scan(":[0-9]{4}")) | ltrimstr(":"),
 #          no_of_hosts: length}'
 #         
 
  # Preprocess CLUSTER_SPEC 
  JQUERY='.cluster.master + .cluster.worker |
         {hosts: map(sub(":[0-9]{4}"; "")), 
          port: (.[0] | scan(":[0-9]{4}")) | ltrimstr(":"),
          no_of_hosts: length}'
         
  CLUSTER_SPEC_PROCESSED=$(echo $CLUSTER_SPEC | jq "$JQUERY")
 
  # Prepare horovodrun parameters
  HOSTS=$(echo "[$CLUSTER_SPEC, $NP]" |
          jq '(":" + (.[1] | tostring)) as $np |
          .[0].cluster.master + .[0].cluster.worker |
          map(sub(":[0-9]{4}"; $np)) |
          join(",")' )
          
  NO_HOSTS=$(echo $CLUSTER_SPEC | 
             jq '.cluster.master + .cluster.worker | 
             length')
             
  PORT=$(echo $CLUSTER_SPEC | 
         jq '.cluster.master[0] |
         scan(":[0-9]{4}") | 
         ltrimstr(":")')
  
  
  echo $HOSTS
  echo $NO_HOSTS
  echo $PORT

  HOROVODRUN="horovodrun -np $NO_HOSTS -H $HOSTS"
  echo $HOROVODRUN
  
  #exec horovodrun -np 1 -H localhost:1 python keras_mnist_advanced.py
else
  echo "worker"
  #exec bash -c ""
fi