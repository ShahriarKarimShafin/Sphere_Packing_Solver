#!/bin/bash
#SBATCH --job-name=PackingOrdered64Cage1Widen
#SBATCH --partition=kamiak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --threads-per-core=1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=packing_cage1_widen_%j.out
#SBATCH --error=packing_cage1_widen_%j.err

set -e

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

module load intel/25.0
module load anaconda3
module load cuda/12.2
source activate dem_env

cd /data/lab/boddeti/sphere_packing_solver

python_bin=$(command -v python)
[[ -x "$python_bin" ]] || { echo "Python is not executable: $python_bin" >&2; exit 1; }

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Concurrent solver cases: 64 (four batches of 16)"
echo "Python executable: $python_bin"
echo "Initializer: ordered; best crystal, cage 1, widening enabled; then none -> hybrid"

ratios=(2to1 3to1 5to1 7to1)
output_root="output/DOE_all4_ordered_cage1_widen_${SLURM_JOB_ID}"

for file in batch.py stages.py refine.py lattice_seeds/binary_lattice.py lattice_seeds/bimodal_lattice.py lattice_seeds/build_seed.py; do
    [[ -s "$file" ]] || { echo "Missing or empty solver file: $file" >&2; exit 1; }
done

shopt -s nullglob
for ratio in "${ratios[@]}"; do
    input_dir="examples/psd/DOE_${ratio}_16_PSDs"
    [[ -d "$input_dir" ]] || { echo "Missing PSD folder: $input_dir" >&2; exit 1; }
    csv_files=("$input_dir"/*.csv)
    [[ ${#csv_files[@]} -eq 16 ]] || {
        echo "Expected 16 CSV files in $input_dir; found ${#csv_files[@]}" >&2
        exit 1
    }
done

# Use the libraries already built for the running job. Do not replace shared
# libraries while another simulation is using this solver installation.
for file in native/energy.so native/moves.so; do
    [[ -s "$file" ]] || {
        echo "Missing native library: $file. Build it before submitting this job, when the shared installation is idle." >&2
        exit 1
    }
done

mkdir -p "$output_root"
pids=()
for ratio in "${ratios[@]}"; do
    batch_output="$output_root/DOE_${ratio}"
    mkdir -p "$batch_output"
    echo "Starting $ratio: 16 cases on 16 cores; log: $batch_output/batch.log"

    # Four exclusive 16-CPU steps use 64 CPUs; four 15G budgets total 60G.
    # Keep this command on one physical line for the cluster's older Slurm.
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --cpu-bind=cores --mem=15G "$python_bin" -u "$PWD/batch.py" --psd-dir "$PWD/examples/psd/DOE_${ratio}_16_PSDs" --output-dir "$batch_output" --workers 16 --expected-cases 16 --initializer ordered --lattice-cage 1 --lattice-widen --lattice-cache "$PWD/$output_root/_lattice_cache" > "$batch_output/batch.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

echo "Job finished at: $(date); batch failure status: $failed"
echo "Results: $(pwd)/$output_root"
exit "$failed"
