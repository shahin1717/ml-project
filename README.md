# Ensemble Methods: Boosting vs. Bagging

This repository contains the source code, experimental pipeline, and documentation for the **Machine Learning Final Project** at the **AI Academy, National AI Center** (Spring 2026).

The project explores a fundamental machine learning question:  
> **"Under what conditions does boosting outperform bagging, and vice versa, and why?"**

To answer this question, we implement key machine learning algorithms **from scratch** (using only NumPy) and compare their performance under various controlled conditions.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Requirements & What We Need to Do](#-requirements--what-we-need-to-do)
  - [1. Module Implementations (From Scratch)](#1-module-implementations-from-scratch)
  - [2. Unsupervised Pipeline (From Scratch)](#2-unsupervised-pipeline-from-scratch)
  - [3. Experimental Design](#3-experimental-design)
  - [4. Deliverables](#4-deliverables)
- [Project Directory Structure](#-project-directory-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Quality Assurance & Constraints](#-quality-assurance--constraints)
- [Grading & Evaluation Rubric](#-grading--evaluation-rubric)

---

## 🔍 Project Overview

The objective is to implement three main supervised learning models:
1. **Decision Tree** (CART algorithm)
2. **AdaBoost** (discrete SAMME with decision stumps)
3. **Random Forest** (bagged trees with feature sub-sampling)

We also implement three unsupervised learning methods:
1. **PCA** (Principal Component Analysis)
2. **K-Means Clustering**
3. **DBSCAN Clustering**

Finally, we conduct a series of controlled experiments evaluating scaling behavior, noise robustness, bias-variance trade-offs, and unsupervised patterns on real-world datasets.

---

## 🎯 Requirements & What We Need to Do

### 1. Module Implementations (From Scratch)
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

### 3. Experimental Design (`experiments/`)
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

### 4. Deliverables
*   **Source Code:** Checked into the GitHub repository under git tag `v1.0-final`.
*   **IEEE-Style Report:** Compiled PDF (`report/report.pdf`), 2-column format, exactly **6–8 pages** long (excluding references).
*   **Professional Slide Deck:** Slide deck PDF (`presentation/presentation.pdf`), 10–12 slides summarizing the project.
*   **Contribution Report:** PDF file (`contribution_report.pdf`) at the repository root outlining what parts of the code, experiments, slides, and report each member did, signed by all members.
*   **Oral Defense:** A 10-minute presentation followed by a 10–12 minute Q&A session.

---

## 📁 Project Directory Structure

```
ml-project/
├── README.md                       # Setup, dependencies, and execution guide
├── requirements.txt                # Pinned python packages
├── .gitignore                      # Ignore large datasets, checkpoints, and compiled TeX
├── download_data.sh                # Bash script to fetch datasets
├── contribution_report.pdf         # Individual contributions report (signed)
├── src/
│   ├── __init__.py
│   ├── trees/
│   │   ├── __init__.py
│   │   └── decision_tree.py        # Decision Tree class & DecisionStump subclass
│   ├── boosting/
│   │   ├── __init__.py
│   │   └── adaboost.py             # AdaBoostClassifier class
│   ├── bagging/
│   │   ├── __init__.py
│   │   └── random_forest.py        # RandomForestClassifier class
│   ├── unsupervised/
│   │   ├── __init__.py
│   │   ├── pca.py                  # Principal Component Analysis
│   │   ├── kmeans.py               # K-Means clustering
│   │   └── dbscan.py               # DBSCAN clustering
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation.py           # Custom evaluation metrics (macro F1, accuracy, etc.)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py        # Standardizer, label encoders, oversampling / SMOTE
│   └── experiments/
│       ├── __init__.py
│       ├── run_all.py              # Main execution script running all experiments
│       └── utils.py                # Dataset loaders & experimental helpers
├── tests/
│   ├── __init__.py
│   ├── test_decision_tree.py       # Unit tests for Decision Tree / Stumps
│   ├── test_adaboost.py            # Unit tests for AdaBoost
│   ├── test_random_forest.py       # Unit tests for Random Forest
│   └── test_unsupervised.py        # Unit tests for PCA, K-Means, DBSCAN
├── notebooks/                      # Development & exploration notebooks (optional bonus)
├── figures/                        # Generated charts & plots for the report
├── data/                           # Downloaded datasets (git-ignored, except script)
├── report/
│   ├── report.tex                  # LaTeX source document
│   └── report.pdf                  # Compiled IEEE-style report
└── presentation/
    └── presentation.pdf            # Oral presentation slide deck
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- bash environment (WSL, Linux, macOS) or equivalent wget utility to run data downloads.

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url> team-name
cd team-name

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

## ⚠️ Quality Assurance & Constraints

To prevent automatic grade deductions, adhere to these strict limits:
1. **No sklearn classifiers:** Do not use `sklearn.tree` or `sklearn.ensemble` classes for primary runs. Doing so incurs a **-25 points** penalty.
2. **Deterministic behavior:** Always use `random_state` seeds for splits, bootstrapping, feature selections, and cluster initializations.
3. **No TODOs / FIXMEs:** Ensure all temporary comments are removed prior to tagging `v1.0-final` (**-3 points** penalty otherwise).
4. **Test Coverage:** Maintain at least **60% unit test coverage** across implemented source modules (**-5 points** penalty otherwise).
5. **No hard-coded paths:** Use relative directories or command line arguments to resolve file locations (**-5 points** penalty otherwise).
6. **Disclose AI tools:** Acknowledge AI utilization (e.g. Copilot, Gemini) in the report's "Tools & Acknowledgements" section.

---

## 🎖️ Advanced Bonuses (Optional)

We can achieve up to **+10 bonus points** by implementing and documenting the following extra credit modules:
*   **Gradient Boosting Machine (GBM) [ +4 pts ]:** Basic GBM implementation with log-loss compared against AdaBoost.
*   **t-SNE Visualization [ +2 pts ]:** Implement or wrap t-SNE for comparisons with PCA.
*   **Multiclass AdaBoost (SAMME.R) [ +2 pts ]:** Real-valued probability variant of AdaBoost for improved multiclass calibration.
*   **GitHub Actions CI [ +1 pt ]:** Set up automatic linting, type-checking (with `mypy`/`pyright`), and test execution with a 60% coverage gate.
*   **Interactive Exploration Notebook [ +1 pt ]:** `notebooks/exploration.ipynb` with interactive widgets (sliders, parameters selector) for visual inspection.
