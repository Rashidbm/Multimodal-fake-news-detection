#!/bin/bash
# Waits for all downloads to complete, then auto-runs the V1 pipeline.
# Safe to start now - it polls every 60 seconds and proceeds when ready.

cd "$(dirname "$0")/.."
LOG=outputs/auto_pipeline.log
mkdir -p outputs
echo "[$(date)] Auto-pipeline started" | tee -a "$LOG"

# --- Wait for downloads to complete ---
echo "[$(date)] Waiting for all downloads to complete..." | tee -a "$LOG"
while true; do
    active=$(ps aux | grep -E 'curl -L|hf_hub_download|load_dataset' | grep -v grep | wc -l | tr -d ' ')
    dgm4_size=$(du -sb data/raw/DGM4 2>/dev/null | awk '{print $1}')
    # Check if all expected files are at correct size
    ready=true
    for f in data/raw/DGM4/origin/*.zip data/raw/DGM4/manipulation/*.zip; do
        if [ -f "$f" ]; then
            # File must have stopped growing (same size check)
            sz1=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")
            sleep 5
            sz2=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")
            if [ "$sz1" != "$sz2" ]; then
                ready=false
                break
            fi
        fi
    done

    if [ "$active" = "0" ] && [ "$ready" = "true" ]; then
        echo "[$(date)] All downloads appear complete." | tee -a "$LOG"
        break
    fi
    echo "[$(date)] active=$active, DGM4=$(du -sh data/raw/DGM4 | awk '{print $1}'), waiting..." | tee -a "$LOG"
    sleep 60
done

# --- Run pipeline ---
echo "[$(date)] Starting V1 pipeline..." | tee -a "$LOG"
bash scripts/run_v1_pipeline.sh 2>&1 | tee -a "$LOG"

echo "[$(date)] Auto-pipeline finished." | tee -a "$LOG"
