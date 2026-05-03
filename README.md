# Cuffless Blood Pressure Estimation from PPG

This repository contains the code I used for my final-year dissertation project on cuffless blood pressure estimation from PPG signals.

The main model is a Siamese neural network. It compares a patient-specific anchor PPG window with a current PPG window, then predicts the change in systolic and diastolic blood pressure. The final system uses PPG-derived waveform channels, spectrogram features, handcrafted beat features, post-hoc calibration, and ensemble averaging.

## Important data note

The MIMIC-II waveform data are not included in this repository. The processed NumPy arrays, cached features, trained checkpoints, and large output folders are also not included.

Access to MIMIC-II must be obtained separately through PhysioNet. After preprocessing, the training and evaluation scripts expect local dataset folders such as `C:\MIMIC2_out_v11`.

This repository is mainly for showing the code used in the project, not for distributing the dataset.

## Main files

These are the files I used most in the final pipeline:

- `preprocess_mimic2_v4.py`  
  Builds the cleaned PPG-only dataset used for the final v11 experiments.

- `preprocess_mimic2_ecgppg.py`  
  Builds the ECG+PPG dataset used for the exploratory hybrid experiment.

- `dataset.py`  
  Loads patient windows, anchor/current pairs, handcrafted features, beat-aware features, and spectrogram inputs.

- `siamese_cnn.py`  
  Defines the Siamese neural network architecture used for BP estimation.

- `train.py`  
  Main training script. It trains one model checkpoint with the selected architecture and training settings.

- `run_ensemble.py`  
  Trains and evaluates the multi-seed ensemble. This was used for the final clean ensemble result.

- `evaluate.py`  
  Evaluates saved checkpoints and reports MAE, RMSE, Pearson correlation, mean error, and BHS grade.

- `patient_specific_adapt.py`  
  Runs the personalised subject-calibrated extension.

- `sweep_bhs_sbp_residual.py`  
  Tests different SBP residual-correction settings.

- `plot_final_v11_dissertation_figures.py`  
  Generates the final figures used in the dissertation for the clean v11 ensemble and personalised extension.

- `plot_training_history.py`  
  Generates training and validation curves from a saved training history.

- `app.py`  
  Streamlit dashboard for the real-time visualisation demo.

There are also older experiment scripts in the repository. I kept them because they show the development process, but the files above are the main ones for the final reported pipeline.

## Why there are multiple preprocessing scripts

The preprocessing scripts are separate because they were used for different experiments:

- `preprocess_mimic2_v4.py` is the final PPG-only preprocessing script. This is the one used for the main v11 result in the dissertation.
- `preprocess_mimic2_ecgppg.py` is separate because the ECG+PPG experiment needed a different input setup with ECG as an extra channel.
- `preprocess_mimic2_v2.py` and `preprocess_mimic2_v3.py` are older development versions. They are kept as legacy files to show the experimentation path, but they are not the final reported pipeline.

I kept the ECG+PPG preprocessing separate rather than merging it into the PPG-only script because the final PPG result and the ECG+PPG negative result were evaluated as different dataset variants.

## Final reported setup

The final clean ensemble used:

- 10-second PPG windows
- PPG, VPG, and APG waveform channels
- STFT spectrogram branch
- handcrafted PPG features
- beat-aware feature stream
- gated pair interaction
- multi-anchor evaluation
- three independently trained seeds
- validation-fitted calibration and SBP residual correction

The strongest clean ensemble result reported in the dissertation was:

- SBP MAE: 6.73 mmHg
- DBP MAE: 3.70 mmHg
- Combined MAE: 5.21 mmHg
- BHS grade: C/A

A personalised subject-calibrated extension was also tested. That achieved a lower combined MAE, but it requires subject-specific calibration data and is therefore reported separately from the clean ensemble result.

## Running the main training script

Example command for one final-style training run:

```bash
python train.py ^
  --data_dir "C:\MIMIC2_out_v11" ^
  --save_dir ".\checkpoints_v11_abpbeat" ^
  --seed 42 ^
  --use_fusion ^
  --use_beat_features ^
  --use_gated_pair_interaction ^
  --no_mixup ^
  --no_balanced_sampling ^
  --sbp_loss_type huber ^
  --ccc_weight 0.06 ^
  --sbp_scale_weight 0.03 ^
  --epochs 150 ^
  --patience 30 ^
  --batch_size 256 ^
  --eval_batch_size 512 ^
  --lr 3e-4 ^
  --dropout 0.30 ^
  --embed_dim 256 ^
  --weight_decay 1e-3 ^
  --num_anchors 5 ^
  --target_scale 10.0 ^
  --quality_threshold 0.4 ^
  --preserve_ppg_amplitude ^
  --multi_anchor_eval ^
  --use_ema ^
  --ema_decay 0.999 ^
  --gap_penalty 0.15 ^
  --gap_target 1.0 ^
  --corr_weight 0.05 ^
  --sbp_weight 1.5 ^
  --tta 0 ^
  --workers 0
```

The exact commands used for individual experiments may differ slightly depending on the run folder.

## Running the ensemble

Example:

```bash
python run_ensemble.py --data_dir "C:\MIMIC2_out_v11" --n_seeds 3 --ensemble_dir ".\ensemble_v11_abpbeat"
```

This trains/evaluates multiple seeds and writes the ensemble outputs to the selected folder.

## Running the visualisation demo

The Streamlit dashboard can be launched with:

```bash
streamlit run app.py
```

The app expects the processed dataset and model checkpoint paths to exist locally. It shows predicted BP, reference BP, error, running MAE, PPG/VPG/APG waveforms, and a BP prediction-history plot.

## What is not included

The following are intentionally not included:

- MIMIC-II waveform data
- processed `.npy` arrays
- cached spectrograms or feature files
- trained model checkpoints
- large output folders
- dissertation Word/PDF drafts
- generated logs

This keeps the repository small and avoids redistributing restricted data.

## Notes

This code was written as a dissertation research project rather than a polished software package. Some older files are kept because they show the path of experimentation. The final reported results are based on the cleaned v11 PPG-only pipeline, the three-seed ensemble, and the separately reported personalised calibration extension.
