#!/bin/sh

FILE_NAME=$1

#chmod -R +x /home/sjladams/projects/concentration_inequalities

cat > temp_job_script.sh <<EOF
#
# requested nr nodes and cores:
#PBS -l nodes=1:ppn=8,mem=32gb
#
# name job:
#PBS -N ${FILE_NAME}
#
# names output and error files:
#PBS -j oe
#PBS -o logging/${FILE_NAME}.pbs.txt

# Ensure the output and error directories exist
# mkdir -p ${PBS_O_HOME}/projects/ConcentrationInequalities/logging

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# setup python environment
echo "TMPDIR='$TMPDIR' PBS_JOBID='${PBS_JOBID:-}'"


cd $PBS_O_HOME
source miniconda3/bin/activate
conda activate concentration_inequalities

cd projects/ConcentrationInequalities
LOG="logging/${FILE_NAME}.txt"
mkdir -p logging

# Time file-transfer time
mkdir -p "${TMPDIR:-/tmp/$PBS_JOBID}"
TMPDIR="${TMPDIR:-/tmp/$PBS_JOBID}"
mkdir -p "$TMPDIR"
vmstat 1 > "$TMPDIR/vmstat.log" &
VMSTAT_PID=$!

# execute scheduler file:
stdbuf -oL -eL bash schedulers/${FILE_NAME}.sh >> "$LOG" 2>&1

kill $VMSTAT_PID
cp "$TMPDIR/vmstat.log" logging/${FILE_NAME}.vmstat.txt

EOF

# submit the job script
qsub temp_job_script.sh