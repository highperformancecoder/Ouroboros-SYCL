#!/usr/bin/bash

echo "Case,NumAllocations,Size,Allocation,Free"
for numAllocations in 256 512 1024 2048 4096 8192; do
    for size in 16 32 64 128 256 512 1024 2048 4096 8192; do
        for c in main_c main_p va_main_c va_main_p vl_main_c vl_main_p; do
            echo -n "$c,$numAllocations,$size,"
            ./$c $numAllocations $size &>log
            if [ $? eq 0 ]; then
                grep "Timing Allocation" log|cut -c20-27|tr -d 's'|tr '\n' ','
                grep "Timing       Free" log|cut -c20-27|tr -d 's'
            else
                echo "aborted"
            fi
            sleep 1
        done
    done
done
