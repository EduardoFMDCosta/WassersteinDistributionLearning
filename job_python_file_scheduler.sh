#!/bin/sh

FILE_NAME=$1

#chmod -R +x /home/sjladams/projects/ConcentrationInequalities

cat > temp_job_script.sh <<EOF
#
# requested nr nodes and cores:
#PBS -l nodes=1
#
# name job:
#PBS -N ${FILE_NAME}
#
# names output and error files:
#PBS -j oe
#PBS -o logging/${FILE_NAME}.txt

# Ensure the output and error directories exist
# mkdir -p ${PBS_O_HOME}/projects/ConcentrationInequalities/logging

# setup python environment
cd $PBS_O_HOME
source miniconda3/bin/activate
conda activate concentration_inequalities

# execute scheduler file:
cd projects/ConcentrationInequalities
python -m ${FILE_NAME}

EOF

# submit the job script
qsub temp_job_script.sh