#!/bin/bash

for i in {2..10} 
#do
#    echo "run_hybrid_wec_mw3_ns${i}.sh"
#    sbatch run_hybrid_wec_mw3_ns${i}.sh
#    sleep 1
#done

#for i in {2..10}
#do
#    echo "run_diam_wec_mw3_ns${i}.sh"
#    sbatch run_diam_wec_mw3_ns${i}.sh
#    sleep 1
#done

for i in {2..10}
do     
    echo "run_angle_wec_mw3_ns${i}.sh"
    sbatch run_angle_wec_mw3_ns${i}.sh
    sleep 1
done
