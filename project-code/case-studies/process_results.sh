#!/bin/bash
for i in {2..10}
do
    #cd snopt_wec_hybrid_max_wec_3_nsteps_$i.000
    #echo snopt_wec_hybrid_max_wec_3_nsteps_$i.000
    #cd snopt_wec_diam_max_wec_4_nsteps_$i.000
    #echo snopt_wec_diam_max_wec_4_nsteps_$i.000
    cd snopt_wec_angle_max_wec_10_nsteps_$i.000
    echo snopt_wec_angle_max_wec_10_nsteps_$i.000
    # rm *all*
    cp ../parse_out_files.py .
    python parse_out_files.py
    git add *all*
    cd ..
done
