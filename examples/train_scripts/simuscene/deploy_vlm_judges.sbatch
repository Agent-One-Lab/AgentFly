#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=120:00:00
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --output=logs/simuscene-judges-%j.log
#SBATCH --error=logs/simuscene-judges-%j.log

# =============================================================================
# Serve the three SimuScene VLM judges, one model per node, all on port 8000.
#
# simuscene_reward.py calls http://<VLM_SERVER_IPS[i]>:8000/v1 with
# model=<VLM_MODELS[i]> and passes the rendered video as a file:// path, so:
#   * every server listens on port 8000 and is served under its HF id;
#   * every server may read local media under the shared filesystem;
#   * the training job's VLM_SHARED_VIDEO_DIR must be on that filesystem.
#
#   cd AgentFly && sbatch examples/train_scripts/simuscene/deploy_vlm_judges.sbatch
#
# When the log prints "ALL JUDGES READY", source the env file it names from the
# training script to get VLM_MODELS / VLM_SERVER_IPS.
# =============================================================================

set -uo pipefail

# Order matters: VLM_MODELS[i] is served by node i.
MODELS=(
    "Qwen/Qwen3-VL-235B-A22B-Instruct"
    "OpenGVLab/InternVL3_5-241B-A28B"
    "zai-org/GLM-4.6V"
)
# Where each model is loaded from (served name stays the HF id above). InternVL's
# remote-code config does a relative import that transformers 5.x resolves next to the
# HF-cache blob instead of the snapshot, so it loads from a copy with real .py/.json files
# and symlinked weights.
SOURCES=(
    "Qwen/Qwen3-VL-235B-A22B-Instruct"
    "/mnt/weka/home/renxi.wang/.cache/simuscene_judges/InternVL3_5-241B-A28B"
    "zai-org/GLM-4.6V"
)
# "<tp> <dp> <extra vllm flags>" per model. GLM-4.6V would fit TP4, but TP4 x DP2 on one
# node hung in CUDA-graph capture (job 2616509), so every judge runs as one TP8 engine.
# GLM's vision tower has 12 heads (not divisible by 8), so its encoder runs data-parallel.
PARALLEL=(
    "8 1 "
    "8 1 --trust-remote-code"
    "8 1 --reasoning-parser glm45 --mm-encoder-tp-mode data"
)

PORT=8000
# Context length per model; InternVL3.5's config caps it at 40960.
MAX_MODEL_LENS=(65536 40960 65536)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MEDIA_ROOT="${MEDIA_ROOT:-/mnt/weka}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-2400}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENGINE_CONDA_ENV="${ENGINE_CONDA_ENV:-vllm-test2}"
export HF_HOME="${HF_HOME:-/mnt/weka/home/renxi.wang/.cache/huggingface}"
export HF_HUB_OFFLINE=1

mkdir -p logs
RUN_DIR="logs/simuscene-judges-${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
ENV_FILE="${RUN_DIR}/judges.env"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if [ "${#nodes[@]}" -lt "${#MODELS[@]}" ]; then
    echo "ERROR: need ${#MODELS[@]} nodes, got ${#nodes[@]}"; exit 1
fi

PIDS=()
IPS=()
for i in "${!MODELS[@]}"; do
    node="${nodes[$i]}"
    model="${MODELS[$i]}"
    source_path="${SOURCES[$i]}"
    max_model_len="${MAX_MODEL_LENS[$i]}"
    read -r tp dp extra <<< "${PARALLEL[$i]}"
    ip=$(getent hosts "${node}" | awk '{print $1}')
    IPS+=("${ip}")
    log="${RUN_DIR}/$(basename "${model}").log"
    echo "Launching ${model} on ${node} (${ip}:${PORT}) tp=${tp} dp=${dp} -> ${log}"

    srun --nodes=1 --ntasks=1 --nodelist="${node}" --exclusive \
         --output="${log}" --error="${log}" \
         bash -c "
            set +u
            source '${CONDA_SH}'
            conda activate '${ENGINE_CONDA_ENV}'
            # Same CUDA-compat / JIT-toolchain workarounds as SimuEval/deploy/serve_multi.sbatch.
            export CUDA_HOME=\"\$CONDA_PREFIX\"
            export PATH=\"\$CONDA_PREFIX/bin:\$PATH\"
            export CC=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc\"
            export CXX=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\"
            export LIBRARY_PATH=\"\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/targets/x86_64-linux/lib/stubs\${LIBRARY_PATH:+:\$LIBRARY_PATH}\"
            export LD_LIBRARY_PATH=\"\$(bash /mnt/weka/home/renxi.wang/Research/deploy/cuda_compat_prefix.sh \"\$CONDA_PREFIX/cuda-compat\")\$CONDA_PREFIX/lib:\$CONDA_PREFIX/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}\"
            cache=/tmp/simuscene-judge-${SLURM_JOB_ID}
            export TRITON_CACHE_DIR=\$cache/triton TORCHINDUCTOR_CACHE_DIR=\$cache/inductor VLLM_CACHE_ROOT=\$cache/vllm
            exec vllm serve '${source_path}' \
                --served-model-name '${model}' \
                --host 0.0.0.0 --port ${PORT} \
                --tensor-parallel-size ${tp} --data-parallel-size ${dp} \
                --dtype bfloat16 \
                --max-model-len ${max_model_len} \
                --gpu-memory-utilization ${GPU_MEM_UTIL} \
                --limit-mm-per-prompt '{\"image\": 0, \"video\": 1}' \
                --allowed-local-media-path '${MEDIA_ROOT}' \
                ${extra}
         " &
    PIDS+=($!)
done

cleanup() {
    echo "[cleanup] stopping judges"
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for ${#MODELS[@]} judges (timeout ${HEALTH_TIMEOUT_SEC}s)..."
t_start=$(date +%s)
while true; do
    ready=0
    for ip in "${IPS[@]}"; do
        curl -sf --max-time 3 "http://${ip}:${PORT}/health" >/dev/null 2>&1 && ready=$(( ready + 1 ))
    done
    [ "${ready}" -eq "${#MODELS[@]}" ] && break
    for p in "${PIDS[@]}"; do
        if ! kill -0 "${p}" 2>/dev/null; then
            echo "ERROR: a judge exited before becoming healthy; see ${RUN_DIR}/*.log"; exit 1
        fi
    done
    if (( $(date +%s) - t_start > HEALTH_TIMEOUT_SEC )); then
        echo "ERROR: only ${ready}/${#MODELS[@]} judges healthy"; exit 1
    fi
    sleep 15
done

models_csv=$(IFS=,; echo "${MODELS[*]}")
ips_csv=$(IFS=,; echo "${IPS[*]}")
cat > "${ENV_FILE}" <<EOF
export VLM_MODELS="${models_csv}"
export VLM_SERVER_IPS="${ips_csv}"
EOF

echo "==================================================================="
echo "ALL JUDGES READY. Source this from the training script:"
echo "  source $(realpath "${ENV_FILE}")"
cat "${ENV_FILE}"
echo "Holding the allocation until --time or scancel ${SLURM_JOB_ID}."
echo "==================================================================="
wait
