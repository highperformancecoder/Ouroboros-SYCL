#!/usr/bin/bash

echo "Case,NumAllocations,Size,Allocation,Subs Allocation,Free,Subs Free"
numAllocations=256
while [ $numAllocations -le 8192 ]; do
    size=1000
    while [ $size -lt 8192 ] ; do
        for c in main_c main_p va_main_c va_main_p vl_main_c vl_main_p; do
            echo -n "$c,$numAllocations,$size,"
            ./$c $numAllocations $size &>log &
            pid=`jobs -p`
            sleep 500&
            wait -n
            if [ $? -eq 0 ]; then
                # possibly still running
                jobs -p>jobs
                if grep $pid jobs>/dev/null; then
                    # still running
                    echo "NaN,NaN,NaN,NaN"
                else
                    grep "Timing," log|cut -c9-
                fi
            else
                echo "NaN,NaN,NaN,NaN"
            fi
            still_running=`jobs -p`
            disown
            for pid in $still_running; do kill $pid; done
            sleep 1
        done
        size=$[size+1000]
    done
    numAllocations=$[numAllocations+256]
done
