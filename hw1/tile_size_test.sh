#!/bin/bash

for ((i = 2; i <= 7; i++)); do
    arg_value=$((2 ** i))

    echo "Testing Runtime of $arg_value"

    ./matrix_multiply -s $arg_value

    echo "Testing CacheGrind of $arg_value"

    valgrind --tool=cachegrind --branch-sim=yes ./matrix_multiply -s $arg_value
    echo
done