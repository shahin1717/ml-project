# Ensemble Methods: Boosting vs. Bagging

This repository contains the source code, experimental pipeline, and documentation for the **Machine Learning Final Project** at the **AI Academy, National AI Center** (Spring 2026).

The project explores a fundamental machine learning question:  
> **"Under what conditions does boosting outperform bagging, and vice versa, and why?"**

To answer this question, we implement key machine learning algorithms **from scratch** (using only NumPy) and compare their performance under various controlled conditions.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Requirements & Specifications](#-requirements--specifications)
  - [1. Supervised Module Implementations (From Scratch)](#1-supervised-module-implementations-from-scratch)
  - [2. Unsupervised Pipeline (From Scratch)](#2-unsupervised-pipeline-from-scratch)
  - [3. Preprocessing & Metrics](#3-preprocessing--metrics)
- [Experimental Design](#-experimental-design)
- [Project Directory Structure](#-project-directory-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Deliverables](#-deliverables)
- [Grading Rubric & Penalty Rules](#-grading-rubric--penalty-rules)
  - [Grading Breakdown](#grading-breakdown)
  - [Automatic Deductions](#automatic-deductions)
  - [Contribution Report Penalties](#contribution-report-penalties)
- [Advanced Bonuses (Optional)](#-advanced-bonuses-optional)

---

## 🔍 Project Overview
- **Released:** June 30, 2026
- **Due Date:** July 16, 2026 at 23:59 (UTC+4)
- **Workload:** ~35–40 person-hours over 12 days
- **Team Size:** 3 students (2 or 4 with approval)
- **Submission:** GitHub repo tag `v1.0-final` + Moodle + Oral Defense
- **Team Registration:** Email group composition and dataset choices to **sultan.musayeva@aiacademy.az** by Day 2 (June 23, 23:59 UTC+4).

The objective is to implement three main supervised learning models:
1. **Decision Tree** (CART algorithm)
2. **AdaBoost** (discrete SAMME with decision stumps)
3. **Random Forest** (bagged trees with feature sub-sampling)

We also implement three unsupervised learning methods to explore and visualize the data:
1. **PCA** (Principal Component Analysis)
2. **K-Means Clustering**
3. **DBSCAN Clustering**

Finally, we conduct a series of controlled experiments evaluating scaling behavior, noise robustness, bias-variance trade-offs, and unsupervised patterns on real-world datasets.

---

## 🎯 Requirements & Specifications

### 1. Supervised Module Implementations (From Scratch)
All supervised and unsupervised algorithms must be written without using scikit-learn's model classes (scikit-learn is allowed *only* as a validation baseline, and for calculating the Adjusted Rand Index).

#### 🌳 Module 1 — Decision Tree Classifier (`src/trees/decision_tree.py`)
- **Binary Split Tree (CART):** Support continuous features only.
- **Criteria:** Impurity reduction based on **Gini Impurity** ($1 - \sum p_c^2$) and **Information Gain/Entropy** ($-\sum p_c \log_2 p_c$).
- **Stopping Criteria:** `max_depth`, `min_samples_split`, pure node (all class labels identical), or identical feature vectors.
- **Weighted Splitting:** Must accept an optional `sample_weight` parameter to compute weighted impurity for AdaBoost boosting updates:
  $$p_c = \frac{\sum_{i: y_i=c} w_i}{\sum_i w_i}$$
- **Features Selection:** Accept `max_features` parameter (useful for Random Forest).
- **Introspection:** Implement `depth`, `n_leaves`, `feature_importances()`, and a readable text-based `__repr__` for trees of depth $\le 4$.

#### 🚀 Module 2 — AdaBoost with Decision Stumps (`src/boosting/adaboost.py`)
- **Stump Base Learner:** Uses `DecisionStump` (a wrapper subclass of `DecisionTree` with `max_depth=1`).
- **Algorithm:** Discrete **SAMME** variant for binary/multi-class classification.
- **Steps:**
  1. Initialize weights $w_i = 1/N$.
  2. For each estimator $m \in \{1, \dots, M\}$:
     - Fit stump $h_m$ using sample weights.
     - Compute weighted error $\epsilon_m$. If $\epsilon_m = 0$, clip to $10^{-10}$. If $\epsilon_m \ge 0.5$, terminate early or document reset behavior.
     - Compute estimator coefficient $\alpha_m = \ln \frac{1 - \epsilon_m}{\epsilon_m} + \ln(K - 1)$ (where $K$ is the number of classes).
     - Update weights $w_i \leftarrow w_i \exp(\alpha_m \cdot \mathbf{1}[h_m(x_i) \neq y_i])$, and normalize.
  3. Final Predict: $\hat{y} = \arg\max_k \sum_{m=1}^M \alpha_m \mathbf{1}[h_m(x) = k]$.
- **Introspection:** Expose `estimator_weights`, `estimator_errors`, and `staged_predict(X)` to get predictions at each step.

#### 🌲 Module 3 — Random Forest Classifier (`src/bagging/random_forest.py`)
- **Bootstrap Sampling:** Draw $N$ samples with replacement.
- **Tree Growing:** Train full `DecisionTree` instances with feature sub-sampling (`max_features` randomly selected at each split node, e.g., $\lfloor\sqrt{p}\rfloor$).
- **Parallelization:** Implement parallel tree training using `multiprocessing.Pool` when `n_jobs > 1`.
- **Out-of-Bag (OOB) Score:** Compute prediction for each sample $i$ using only the trees where sample $i$ was out-of-bag (not selected in the bootstrap sample). Report average accuracy as `oob_score_`.
- **Feature Importances:** Average feature importances across all trees in the forest.

---

### 2. Unsupervised Pipeline (From Scratch)
Used to analyze and visualize datasets before and after classification.

#### 📉 PCA (`src/unsupervised/pca.py`)
- Eigen-decomposition of the covariance matrix.
- Expose components (`components_`) and explained variance ratio (`explained_variance_ratio_`).
- Project data into lower dimensions using `transform(X)`.

#### 🧬 K-Means (`src/unsupervised/kmeans.py`)
- Lloyd's algorithm with multiple random restarts (keeping centroid configuration with lowest inertia).
- Expose final `centroids_`, point `labels_`, and final `inertia_` (sum of squared distances).

#### 🌐 DBSCAN (`src/unsupervised/dbscan.py`)
- Density-based clustering that expands clusters from core points based on neighborhood radius $\varepsilon$ (`eps`) and minimum samples (`min_samples`).
- Identify and mark noise points with label `-1`.

---

### 3. Preprocessing & Metrics
- **Preprocessing (`src/utils/preprocessing.py`):** Custom implementations of `StandardScaler`, `LabelEncoder`, `train_test_split`, and class balancing (oversampling/SMOTE).
- **Evaluation Metrics (`src/metrics/evaluation.py`):** Custom metrics for `accuracy_score`, `confusion_matrix`, and `precision_recall_f1` (macro F1).

---

## 🎨 Experimental Design

All experiments must be automated and fully reproducible via a single script: `python src/experiments/run_all.py`.

1. **Baseline Comparison:**
   - Custom `DecisionTree` vs. custom `DecisionStump` vs. `sklearn.tree.DecisionTreeClassifier`.
   - Verify accuracy difference is within 2%.
2. **AdaBoost Scaling:**
   - Vary $M$ (estimators) from 1 to 200.
   - Record and plot training and testing errors vs. $M$ using `staged_predict`.
3. **Random Forest Scaling:**
   - (a) Fix `max_depth=None`, vary $T$ (estimators) from 1 to 200. Plot test accuracy.
   - (b) Fix $T=100$, vary `max_depth` from 1 to 20. Plot test accuracy and OOB score.
4. **Head-to-Head Comparison (Fixed Resources):**
   - 5-fold cross-validation on 100 estimators.
   - Report Mean $\pm$ SD for single tree, AdaBoost, Random Forest, and `sklearn.ensemble.RandomForestClassifier`.
5. **Noise Robustness:**
   - Corrupt training labels by flipping a fraction $\eta \in \{0.05, 0.10, 0.20\}$ of labels.
   - Train AdaBoost and Random Forest on corrupted labels and evaluate accuracy degradation on clean test data.
6. **Bias-Variance Decomposition:**
   - Generate $B=100$ bootstrap replicates of a balanced binary dataset.
   - Train AdaBoost and Random Forest on each.
   - Calculate Bias$^2$ and Variance on a large test set and compare.
7. **Unsupervised Analysis:**
   - Scree plot (explained variance vs. PCs).
   - PCA 2D scatter plots colored by: True labels, K-Means clusters, and DBSCAN clusters.
   - Report Adjusted Rand Index (ARI) for K-Means and DBSCAN.
   - Generate a k-distance plot to justify the choice of DBSCAN parameter $\varepsilon$.

---

## 📁 Project Directory Structure

```
ml-project/
├── README.md                       # Setup, dependencies, and execution guide
├── requirements.txt                # Pinned python packages
├── .gitignore                      # Git ignore patterns
├── download_data.sh                # Bash script to fetch datasets
├── download_data.py                # Python script to download datasets
├── contribution_report.pdf         # Individual contributions report (signed)
├── experiment_runs.log             # Reproducible execution logs of all experiments
├── src/                            # From-scratch implementations source code
│   ├── __init__.py
│   ├── trees/
│   │   ├── __init__.py
│   │   └── decision_tree.py        # Decision Tree class & DecisionStump subclass
│   ├── boosting/
│   │   ├── __init__.py
│   │   ├── adaboost.py             # AdaBoostClassifier class (SAMME & SAMME.R)
│   │   └── gradient_boosting.py    # Custom Gradient Boosting Classifier
│   ├── bagging/
│   │   ├── __init__.py
│   │   └── random_forest.py        # RandomForestClassifier class (parallelized)
│   ├── unsupervised/
│   │   ├── __init__.py
│   │   ├── pca.py                  # Principal Component Analysis
│   │   ├── kmeans.py               # K-Means clustering
│   │   ├── dbscan.py               # DBSCAN clustering
│   │   └── tsne_comparison.py      # t-SNE evaluation module
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation.py           # Custom evaluation metrics (macro F1, accuracy, etc.)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py        # StandardScaler, LabelEncoder, train_test_split
│   └── experiments/
│       ├── __init__.py
│       └── run_all.py              # Main execution script orchestrating all experiments
├── experiments/                    # Core experiment script modules
│   ├── utils.py                    # Dataset loading & plot helpers
│   ├── scaling.py                  # Experiment 1 baseline comparison
│   ├── adaboost_scaling.py         # Experiment 2 AdaBoost scaling
│   ├── rf_scaling.py               # Experiment 3 Random Forest scaling (estimators & depth)
│   ├── head_to_head.py             # Experiment 4 cross-validation head-to-head comparison
│   ├── noise_robustness.py         # Experiment 5 label noise robustness evaluation
│   ├── bias_variance.py            # Experiment 6 quantitative bias-variance decomposition
│   ├── unsupervised_analysis.py    # Experiment 7 PCA, K-Means, and DBSCAN pipeline
│   ├── gbm_vs_adaboost.py          # Bonus: Custom Gradient Boosting vs AdaBoost
│   ├── sammer_comparison.py        # Bonus: SAMME vs SAMME.R calibration
│   ├── verify_split.py             # Utility to verify Covertype stratified train/test split
│   └── tsne_all_datasets.py        # t-SNE visualization script
├── tests/                          # Automated unit test suite (91% total coverage)
│   ├── __init__.py
│   ├── test_decision_tree.py       # CART tree and stumps tests
│   ├── test_adaboost.py            # AdaBoost classifier tests
│   ├── test_random_forest.py       # Random Forest classifier tests
│   ├── test_gradient_boosting.py   # Gradient Boosting classifier tests
│   ├── test_pca.py                 # PCA tests
│   ├── test_kmeans.py              # K-Means tests
│   ├── test_dbscan.py              # DBSCAN tests
│   ├── test_preprocessing.py       # Standardizer and splitter tests
│   ├── test_evaluation.py          # Custom metrics tests
│   ├── test_bias_variance.py       # Bias-variance decomposition tests
│   └── test_tsne_comparison.py     # t-SNE helper tests
├── slides/                         # Defense materials & explanation guides (git-ignored)
│   ├── defense_slides.tex          # LaTeX presentation slides
│   ├── explanation.md              # Unified QA explanation guide
│   ├── code_explanation.md         # Comprehensive mathematical codebase documentation
│   └── script.md                   # Presentation oral defense script
├── figures/                        # Generated experimental plots & charts for report
├── data/                           # Downloaded datasets (git-ignored)
├── report/
│   ├── report.tex                  # LaTeX source report document
│   └── report.pdf                  # Compiled IEEE-style PDF report
└── presentation/
    └── presentation.pdf            # Compiled Beamer slide deck PDF
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- bash environment (WSL, Linux, macOS) or equivalent wget utility to run data downloads.

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/shahin1717/ml-project
cd ml-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Download Datasets
Execute the shell script to pull data files from original UCI/scikit-learn repositories:
```bash
bash download_data.sh
```

### 2. Run All Experiments
To run the full experimental suite, execute:
```bash
python src/experiments/run_all.py
```
This generates all figures and tables in the `./figures/` and `./report/` directories.

### 3. Run the Test Suite
Ensure the implementations match reference packages and work correctly on edge cases:
```bash
pytest --cov=src tests/
```

---

## 📦 Deliverables
1. **GitHub Repository:** Tagged `v1.0-final`. Contains code, tests, configs, one-command execution.
2. **IEEE-Style Paper (`report/report.pdf`):** 2-column format, exactly **6–8 pages** long (excluding references).
3. **Slide Deck (`presentation/presentation.pdf`):** 10–12 slides summarizing key aspects.
4. **Contribution Report (`contribution_report.pdf`):** Signed report in the root outlining what parts of the code, report, slides, and experiments each member did.
5. **Oral Defense:** 10-minute presentation followed by a 10–12 minute Q&A session.

---

## 🎖️ Grading Rubric & Penalty Rules

### Grading Breakdown
Total project value: **100 points**
*   **Code (45%):**
    *   Correctness of implementations (18%) — within 2% of sklearn, no sklearn ensemble classes used.
    *   Code quality & design (9%) — OOP, type hints, docstrings, no magic numbers.
    *   Unsupervised pipeline (9%) — PCA, K-Means, DBSCAN correct, with visualizations.
    *   Testing (9%) — pytest coverage $\ge 60\%$, seeded random operations.
*   **Report (25%):**
    *   Experimental rigor (10%) — 7 experiments, cross-validation, bias-variance.
    *   Conceptual depth (9%) — correct math explanations (e.g. AdaBoost weight updates).
    *   Clarity & format (6%) — IEEE format, page limits, citations.
*   **Presentation (10%):**
    *   Slide design (3%) — layout, theme, fonts.
    *   Data visualization (3%) — clear figures and labels.
    *   Content organization (4%) — flow, covers all key aspects.
*   **Oral Defense (20%):**
    *   Individual understanding (12%) — ability to defend any part of code/math.
    *   Depth of analysis (5%) — reasoning about failure modes/tradeoffs.
    *   Clarity under pressure (3%) — handling Q&A questions.

### Automatic Deductions
*   Using `sklearn.tree` or `sklearn.ensemble` classes for primary runs: **-25 points**.
*   Team member unable to explain their own module during defense: **-8 points per module**.
*   Hard-coded file paths or magic constants: **-5 points**.
*   Test coverage below 60%: **-5 points**.
*   `TODO` or `FIXME` comments left in final code: **-3 points**.
*   Missing `run_all.py` or non-reproducible figures: **-5 points**.
*   Report outside the 6–8 page limit: **-3 points**.
*   Missing slide deck or late submission: **-5 points**.
*   Missing contribution report or late submission: **-5 points** (team deduction).

### Contribution Report Penalties
*   A member who cannot substantiate their claimed contributions: **-8 points**.
*   Severe contribution imbalances (e.g., one member doing >70% of the work) will result in individual grade reductions.

---

## 🎖️ Advanced Bonuses (Optional)
We can achieve up to **+10 bonus points** by implementing and documenting the following extra credit modules:
*   **Gradient Boosting Machine (GBM) [ +4 pts ]:** Basic GBM implementation with log-loss compared against AdaBoost.
*   **t-SNE Visualization [ +2 pts ]:** Implement or wrap t-SNE for comparisons with PCA.
*   **Multiclass AdaBoost (SAMME.R) [ +2 pts ]:** Real-valued probability variant of AdaBoost for improved multiclass calibration.
*   **GitHub Actions CI [ +1 pt ]:** Set up automatic linting, type-checking (with `mypy`/`pyright`), and test execution with a 60% coverage gate.
*   **Interactive Exploration Notebook [ +1 pt ]:** `notebooks/exploration.ipynb` with interactive widgets (sliders, parameters selector) for visual inspection.
