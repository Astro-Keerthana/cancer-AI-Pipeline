#!/usr/bin/env nextflow
/*
 * Pipeline: End-to-End Cancer ML Pipeline
 * Author:   Your Name
 * Dataset:  TCGA-BRCA (Breast Cancer)
 * Nextflow: DSL2
 */

nextflow.enable.dsl = 2

// ─── Parameters ───────────────────────────────────────────────
params.max_files   = 20
params.output_dir  = "$projectDir/results"
params.model_dir   = "$projectDir/models"

// ─── Process 1: Data Retrieval ────────────────────────────────
process RETRIEVE_DATA {
    tag "GDC-API"
    publishDir "${params.output_dir}/raw", mode: 'copy'

    output:
    path "metadata.csv",        emit: metadata
    path "tcga_brca_raw.tar.gz", emit: archive

    script:
    """
    python3 ${projectDir}/scripts/01_retrieve_data.py \
        --max_files ${params.max_files} \
        --output_dir .
    """
}

// ─── Process 2: Preprocessing ─────────────────────────────────
process PREPROCESS {
    tag "Preprocessing"
    publishDir "${params.output_dir}/processed", mode: 'copy'

    input:
    path metadata
    path archive

    output:
    path "X_features.csv", emit: features
    path "y_labels.csv",   emit: labels
    path "X_train.npy",    emit: X_train
    path "X_test.npy",     emit: X_test
    path "y_train.npy",    emit: y_train
    path "y_test.npy",     emit: y_test

    script:
    """
    python3 ${projectDir}/scripts/02_preprocess.py \
        --raw_dir . \
        --output_dir .
    """
}

// ─── Process 3: Model Training ────────────────────────────────
process TRAIN_MODEL {
    tag "XGBoost-Training"
    publishDir "${params.model_dir}", mode: 'copy'

    input:
    path X_train
    path X_test
    path y_train
    path y_test
    path features

    output:
    path "xgb_cancer_model.pkl", emit: model
    path "biomarkers.csv",       emit: biomarkers
    path "confusion_matrix.png", emit: cm_plot
    path "biomarkers.png",       emit: feat_plot
    path "metrics.json",         emit: metrics

    script:
    """
    python3 ${projectDir}/scripts/03_train_model.py \
        --processed_dir . \
        --output_dir .
    """
}

// ─── Process 4: Model Evaluation Report ──────────────────────
process GENERATE_REPORT {
    tag "Report"
    publishDir "${params.output_dir}/report", mode: 'copy'

    input:
    path metrics
    path biomarkers
    path cm_plot

    output:
    path "report.html"

    script:
    """
    python3 ${projectDir}/scripts/04_generate_report.py \
        --metrics   ${metrics} \
        --biomarkers ${biomarkers} \
        --cm_plot   ${cm_plot} \
        --output    report.html
    """
}

// ─── Workflow ─────────────────────────────────────────────────
workflow {
    // Step 1: Retrieve
    RETRIEVE_DATA()

    // Step 2: Preprocess
    PREPROCESS(
        RETRIEVE_DATA.out.metadata,
        RETRIEVE_DATA.out.archive
    )

    // Step 3: Train
    TRAIN_MODEL(
        PREPROCESS.out.X_train,
        PREPROCESS.out.X_test,
        PREPROCESS.out.y_train,
        PREPROCESS.out.y_test,
        PREPROCESS.out.features
    )

    // Step 4: Report
    GENERATE_REPORT(
        TRAIN_MODEL.out.metrics,
        TRAIN_MODEL.out.biomarkers,
        TRAIN_MODEL.out.cm_plot
    )
}
