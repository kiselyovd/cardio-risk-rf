# Cardiovascular Disease Classification

Classify the **presence of cardiovascular disease at medical examination** from the Sulianova dataset's 11 features. LightGBM is the main estimator; RandomForest with median imputation is the baseline. The historical repository name is retained.

[Model and weights](https://huggingface.co/kiselyovd/cardio-risk-rf) · [Evaluation protocol](EVALUATION.md) · [Русский](https://github.com/kiselyovd/cardio-risk-rf/blob/main/README.ru.md)

## Results

<!-- ARTIFACT-MANIFEST:START -->
### Current artifact and evaluation

**Replayed on the original 10501-row test fold, legacy seed42 protocol**

| Metric | Value |
|---|---:|
| LightGBM ROC-AUC | 79.77% |
| LightGBM average precision | 78.11% |
| RF ROC-AUC | 79.49% |
| RF average precision | 77.94% |

Target is CVD presence at examination. The exact legacy split is 48999/10500/10501; original training row membership is unrecorded. Average precision uses sklearn AP.

[Evaluation record](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/replay-native.json) · [Pinned model](https://huggingface.co/kiselyovd/cardio-risk-rf/tree/a0117e88461cf4197a2a604203f68dde23fd069a)
<!-- ARTIFACT-MANIFEST:END -->

The source CSV, exact row split, model hashes and prediction arrays accompany the [new replay](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/replay-native.json). Average precision is the sklearn AP measure; thresholds use 0.5. This is a retrospective research classifier, with no prospective ten-year horizon or clinical validation.

## Inspect the model

![Current model SHAP associations](images/shap_summary.png)

[Run the published artifact](https://github.com/kiselyovd/cardio-risk-rf#run-the-published-artifact) · [Data and preprocessing contract](EVALUATION.md) · [Component licenses](LICENSES.md)
