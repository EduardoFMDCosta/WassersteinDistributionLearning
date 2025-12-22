#!/bin/sh

DATASET_NAME=$1
TASK_NAME=$2

#chmod -R +x /home/sjladams/projects/concentration_inequalities

cat > temp_job_script.sh <<EOF
#
# requested nr nodes and cores:
#PBS -l nodes=1
#
# name job:
#PBS -N ${DATASET_NAME}_${TASK_NAME}
#
# names output and error files:
#PBS -o logging/${DATASET_NAME}_${TASK_NAME}_out
#PBS -e logging/${DATASET_NAME}_${TASK_NAME}_err

# Ensure the output and error directories exist
# mkdir -p ${PBS_O_HOME}/projects/ConcentrationInequalities/logging

# setup python environment
cd $PBS_O_HOME
source miniconda3/bin/activate
conda activate concentration_inequalities

# execute scheduler file:
cd projects/ConcentrationInequalities
bash schedulers/${DATASET_NAME}/${TASK_NAME}.sh

EOF

# submit the job script
qsub temp_job_script.sh