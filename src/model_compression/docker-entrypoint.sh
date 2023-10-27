#!/bin/bash

echo "Container is running!"

args="$@"
echo $args



python $args > /dev/null
