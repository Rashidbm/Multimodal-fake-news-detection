#!/bin/bash
# Full V1 pipeline: extract → build dataset → train → evaluate → LLaVA logits
# Run after all downloads are complete.

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

LOG_DIR=outputs/v1_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "=== V1 Pipeline Start ==="

# -----------------------------------------------------------
# Step 1: Extract DGM4 zips
# -----------------------------------------------------------
log "Extracting DGM4 origin zips..."
cd data/raw/DGM4/origin
for z in *.zip; do
    if [ ! -d "${z%.zip}" ]; then
        log "  unzipping $z"
        unzip -q -n "$z"
    fi
done

log "Extracting DGM4 manipulation zips..."
cd ../manipulation
for z in *.zip; do
    if [ ! -d "${z%.zip}" ]; then
        log "  unzipping $z"
        unzip -q -n "$z"
    fi
done

cd "$ROOT"

# -----------------------------------------------------------
# Step 2: Extract MMFakeBench images
# -----------------------------------------------------------
log "Extracting MMFakeBench val images..."
cd data/raw/MMFakeBench/images
for z in MMFakeBench_*.zip; do
    if [ -f "$z" ] && [ ! -d "${z%.zip}" ]; then
        log "  unzipping $z"
        unzip -q -n "$z"
    fi
done
cd "$ROOT"

# -----------------------------------------------------------
# Step 3: Extract NewsCLIPpings
# -----------------------------------------------------------
log "Extracting NewsCLIPpings..."
cd data/raw/NewsCLIPpings
for z in *.zip; do
    if [ -f "$z" ] && [ ! -d "${z%.zip}" ]; then
        log "  unzipping $z"
        unzip -q -n "$z"
    fi
done
cd "$ROOT"

# -----------------------------------------------------------
# Step 4: Build balanced dataset from DGM4
# -----------------------------------------------------------
log "Building balanced training dataset (DGM4 only for V1)..."
python scripts/build_balanced_dataset.py \
    --dgm4 data/raw/DGM4 \
    --output data/processed/balanced_dataset.csv 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------
# Step 5: Train FND-CLIP
# -----------------------------------------------------------
log "Training FND-CLIP on MPS..."
python src/train.py --config config/v1.yaml 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------
# Step 6: Evaluate on MMFakeBench (transfer benchmark)
# -----------------------------------------------------------
log "Evaluating FND-CLIP on MMFakeBench val (transfer)..."
python src/evaluate_transfer.py \
    --checkpoint outputs/v1_fnd_clip/best.pt \
    --mmfakebench data/raw/MMFakeBench \
    --split val \
    --output "$LOG_DIR/fndclip_mmfb_val.yaml" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------
# Step 7: Run LLaVA zero-shot baseline on MMFakeBench
# -----------------------------------------------------------
log "Running LLaVA zero-shot on MMFakeBench val..."
python scripts/extract_llava_logits.py \
    --input data/raw/MMFakeBench/liuxuannan___mm_fake_bench/MMFakeBench_val \
    --images data/raw/MMFakeBench/images/MMFakeBench_val \
    --output "$LOG_DIR/llava_logits_val.csv" \
    --limit 500 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------
# Step 8: Final report
# -----------------------------------------------------------
log "Generating final report..."
python scripts/compile_results.py \
    --fndclip-metrics outputs/v1_fnd_clip/test_metrics.yaml \
    --fndclip-transfer "$LOG_DIR/fndclip_mmfb_val.yaml" \
    --llava-csv "$LOG_DIR/llava_logits_val.csv" \
    --output "$LOG_DIR/v1_results.md" 2>&1 | tee -a "$LOG"

log "=== V1 Pipeline Complete ==="
log "Results: $LOG_DIR/v1_results.md"
