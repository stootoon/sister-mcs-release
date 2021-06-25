#!/bin/bash
# Generates the jobs for comparing x_final to x_opt in all the
cd $SISTER_MCS_DATA
ls | grep "^sweep" | xargs -n1 -I@ sh -c 'python $CFDGITPY/cmd2job.py --jobname=@ --submit "cd $SISTER_MCS_DATA; rm *.cmp; python $SISTER_MCS/code/compute_x_MAP.py --folder=@"; mv job.sh job_cmp_@.sh'
