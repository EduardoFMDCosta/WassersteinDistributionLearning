#!/bin/sh

FILE_NAME=$1

#chmod -R +x /home/sjladams/projects/concentration_inequalities

cat > temp_job_script.sh <<'EOF'
#!/bin/sh
set -eu

# requested nr nodes and cores:
#PBS -l nodes=1:ppn=8,mem=32gb

# name job:
#PBS -N ${FILE_NAME}

# names output and error files:
#PBS -j oe
#PBS -o logging/${FILE_NAME}.pbs.txt

# Ensure the output and error directories exist
# mkdir -p ${PBS_O_HOME}/projects/ConcentrationInequalities/logging

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_PROC_BIND=true
# export OMP_PLACES=cores

# setup python environment
cd $PBS_O_HOME
source miniconda3/bin/activate
conda activate concentration_inequalities

# go to project
cd "$PBS_O_HOME/projects/ConcentrationInequalities"
mkdir -p logging
LOG="logging/${FILE_NAME}.txt"

# --- vmstat logging on node-local disk ---
JOBID="${PBS_JOBID:-__JOBNAME__.$$}"
TMPDIR="/var/tmp/$JOBID"
mkdir -p "$TMPDIR"

vmstat 1 > "$TMPDIR/vmstat.log" &
VMSTAT_PID=$!
# --- end vmstat logging ---

# execute scheduler file:
stdbuf -oL -eL bash "schedulers/__JOBNAME__.sh" >> "$LOG" 2>&1

# cleanup + copy back
kill "$VMSTAT_PID" 2>/dev/null || true
cp "$TMPDIR/vmstat.log" "logging/__JOBNAME__.vmstat.txt" || true
rm -rf "$TMPDIR" 2>/dev/null || true

EOF

# Replace placeholders with the job name (FILE_NAME) in the generated script
# (sed is safe here because __JOBNAME__ is a fixed token)
sed -i "s/__JOBNAME__/${FILE_NAME}/g" temp_job_script.sh

# submit the job script, also export FILE_NAME (optional, but harmless)
qsub -v FILE_NAME="${FILE_NAME}" temp_job_script.sh
