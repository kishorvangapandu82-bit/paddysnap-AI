# ============================================================
# PADDYSNAP AI -- FULL RETRAIN + UPDATE PIPELINE
# Trains all 5 models for 25 epochs (no label smoothing)
# Then regenerates all metrics, graphs, ONNX, reports
# ============================================================

$PYTHON = ".\venv\Scripts\python.exe"
$LOG    = ".\pipeline_log.txt"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

Set-Content -Path $LOG -Value "PADDYSNAP AI -- FULL PIPELINE STARTED: $(Get-Date)"

# ============================================================
# PHASE 1: RETRAIN ALL 5 MODELS (25 EPOCHS EACH)
# ============================================================
Log "====== PHASE 1: TRAINING ======"

$models = @("efficientnet_v2_s", "convnext_tiny", "resnet50", "densenet121", "custom_cnn")

foreach ($model in $models) {
    Log ">>> Training: $model (25 epochs, batch=32) ..."
    & $PYTHON src/train.py --model $model --epochs 25 --batch_size 32 2>&1 | Tee-Object -Append -FilePath $LOG
    if ($LASTEXITCODE -ne 0) {
        Log "ERROR: Training failed for $model. Exiting."
        exit 1
    }
    Log "<<< Done: $model"
}

Log "====== PHASE 1 COMPLETE: All 5 models trained ======"

# ============================================================
# PHASE 2: EVALUATE ALL MODELS ON TEST SET
# ============================================================
Log "====== PHASE 2: EVALUATION ======"
& $PYTHON src/evaluate.py --model all 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 2 COMPLETE ======"

# ============================================================
# PHASE 3: PLOT TRAINING CURVES
# ============================================================
Log "====== PHASE 3: PLOTTING TRAINING CURVES ======"
& $PYTHON src/plot_training.py --model all 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 3 COMPLETE ======"

# ============================================================
# PHASE 4: EXPORT ALL MODELS TO ONNX + INT8
# ============================================================
Log "====== PHASE 4: ONNX EXPORT + INT8 QUANTIZATION ======"
& $PYTHON src/export_onnx.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 4 COMPLETE ======"

# ============================================================
# PHASE 5: ONNX FP32 vs INT8 BENCHMARK COMPARISON
# ============================================================
Log "====== PHASE 5: ONNX BENCHMARK ======"
& $PYTHON src/compare_onnx_test.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 5 COMPLETE ======"

# ============================================================
# PHASE 6: GENERATE VISUAL GRAPHS & INFOGRAPHICS
# ============================================================
Log "====== PHASE 6: GENERATING GRAPHS ======"
& $PYTHON src/plot_onnx_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG
& $PYTHON src/generate_efficientnet_comparison_png.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 6 COMPLETE ======"

# ============================================================
# PHASE 7: BATCH PREDICTIONS + VISUAL OVERLAY GRIDS
# ============================================================
Log "====== PHASE 7: BATCH PREDICTIONS (20 images) ======"
& $PYTHON src/predict_batch_overlay.py --num 20 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 7 COMPLETE ======"

# ============================================================
# PHASE 8: GENERATE HTML REPORT
# ============================================================
Log "====== PHASE 8: HTML REPORT ======"
& $PYTHON src/generate_html_report.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 8 COMPLETE ======"

# ============================================================
# PHASE 9: GENERATE SUBMISSION
# ============================================================
Log "====== PHASE 9: GENERATE SUBMISSION CSV ======"
& $PYTHON src/generate_submission.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 9 COMPLETE ======"

# ============================================================
# PHASE 10: RUN FULL SYSTEM TEST SUITE
# ============================================================
Log "====== PHASE 10: SYSTEM TEST SUITE ======"
& $PYTHON src/test_system.py 2>&1 | Tee-Object -Append -FilePath $LOG
Log "====== PHASE 10 COMPLETE ======"

# ============================================================
# DONE
# ============================================================
Log ""
Log "============================================================"
Log " FULL PIPELINE COMPLETE! All models retrained + updated."
Log " Check pipeline_log.txt for full output."
Log "============================================================"
