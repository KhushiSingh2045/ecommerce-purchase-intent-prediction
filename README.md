# A Comparative Machine Learning and Deep Learning Framework for E-Commerce Purchase Intent Prediction from User Interaction Data

**Status: Phase 1 (project setup) complete. Phase 2 (data collection & understanding) not yet started.**

## Research Question
Does preserving the sequence of user interactions provide better purchase-intent prediction than using only aggregated session-level behavioral features?

## Datasets

| Dataset | Role | Source |
|---|---|---|
| UCI Online Shoppers Purchasing Intention | Aggregated-feature path (Path A) | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| YOOCHOOSE / RecSys Challenge 2015 | Sequential-feature path (Path B) | Kaggle - search "yoochoose" (free account required) |

Place downloaded files in `data/raw/uci_shoppers/` and `data/raw/yoochoose/` respectively. These folders are git-ignored — only the code that produces/consumes them is tracked.

## Project Phases
1. **Project setup** ✅ (this skeleton)
2. Data collection & understanding
3. Data preprocessing
4. Feature engineering (aggregated + sequential)
5. Baseline model (Logistic Regression)
6. Machine learning models (XGBoost, LightGBM)
7. Deep learning models (LSTM, GRU, Attention)
8. Training & testing
9. Comparative evaluation
10. Early purchase-intent prediction
11. Anonymous-user prediction
12. Real-time prediction simulation
13. SHAP explainability
14. Cross-dataset validation
15. Behavioral drift analysis
16. Final results
17. Hypothesis evaluation (H0 vs H1)
18. Final research output

Each notebook in `notebooks/` is a placeholder tagged with which phase it belongs to — they'll be filled in one phase at a time, in order, with review between phases.

## Setup
```bash
pip install -r requirements.txt
```
Kaggle API key (`kaggle.json`) will be needed in Phase 2 for the YOOCHOOSE download — not required yet.
