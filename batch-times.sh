#!/usr/bin/bash

echo "Case,NumAllocations,Size,Allocation,Free"
numAllocations=256
while [ $numAllocations -le 8192 ]; do
    size=1000
    while [ $size -lt 8192 ] ; do
        for c in main_c main_p va_main_c va_main_p vl_main_c vl_main_p; do
            echo -n "$c,$numAllocations,$size,"
            ./$c $numAllocations $size &>log
            if [ $? -eq 0 ]; then
                grep "Timing Allocation" log|cut -c20-27|tr -d '\nms'
                echo -n ','
                grep "Timing       Free" log|cut -c20-27|tr -d '\nms'
                echo ""
            else
                echo "NaN,NaN"
            fi
            sleep 1
        done
        size=$[size+1000]
    done
    numAllocation=$[numAllocation+256]
done
