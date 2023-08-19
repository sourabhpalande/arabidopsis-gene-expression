#!/bin/bash
# datatypes=("arabidopsis" "angiosperm");
#datatypes=("arabidopsis");
datatypes=("angiosperm");
modeltypes=("SVC" "KNN" "MLP" "HGB" "RF");
for d in ${datatypes[@]};
do
    for m in ${modeltypes[@]};
    do
        myjobname=$(echo $d)_$(echo $m);
        echo "Data Type: $d";
        echo "Model Type: $m";
        echo "run_$(echo $d)_ml.py";
        echo "$myjobname.log";
        sbatch --job-name=$myjobname --output=$myjobname.SLURMout \
        --export=DATA=$d,MODEL=$m run_ml_slurm_job.sb;
    done;
done;
