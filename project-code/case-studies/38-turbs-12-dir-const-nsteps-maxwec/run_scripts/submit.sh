#!/bin/bash

#for i in {2..10} 
#do     
#    echo "run_hybrid_wec_mw${i}_ns6.sh"
#    sbatch run_hybrid_wec_mw${i}_ns6.sh
#    sleep 1
#done

#for i in {2..10} 
#do     
#    echo "run_diam_wec_mw${i}_ns6.sh"
#    sbatch run_diam_wec_mw${i}_ns6.sh
#    sleep 1
#done

for i in {40..80..5} 
do     
    echo "run_angle_wec_mw${i}_ns6.sh"
    sbatch run_angle_wec_mw${i}_ns6.sh
    sleep 1
done
