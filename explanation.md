# Magenta Customer Upselling Prediction Project

This project implements a complete machine learning pipeline for predicting customer upselling opportunities. The goal is to identify customers likely to upgrade their services, enabling targeted marketing campaigns and improved revenue generation.

## Project Overview

The dataset contains highly imbalanced binary classification data with only **7.05% positive upselling cases**. The project follows a systematic approach from data exploration through production deployment, ultimately selecting **Random Forest as the final model due to its superior recall performance**.

## Notebook Descriptions

### 0.data-exploration.ipynb
**Purpose**: Initial data exploration and business understanding

**Key Activities**:
- Explores three main datasets: `core_data` (customer demographics), `usage_info` (monthly usage patterns), and `customer_interactions` (support/sales interactions)
- Analyzes customer portfolio structure and upselling distribution across segments
- Identifies data quality issues including missing values in `available_gb` field
- Generates business insights about customer behavior patterns
- Establishes baseline understanding of the 7.05% upselling rate

**Key Insights**: The data reveals significant class imbalance and missing value patterns that require careful handling in subsequent analysis.

### 1a.features-engineering.ipynb
**Purpose**: Comprehensive feature engineering with advanced imputation

**Key Activities**:
- Implements missing value imputation using Lasso regression for `available_gb`
- Aggregates usage patterns over time windows (3, 6, 12 months)
- Creates trend features and usage growth indicators
- Pivots customer interaction data into feature format
- Engineers temporal features and customer lifecycle indicators
- Performs extensive correlation analysis between features
- Implements network-based feature selection to remove highly correlated variables

**Key Insights**: Imputation techniques preserve data relationships better than simple mean imputation, creating more informative features for model training.

### 1b.features-engineering.ipynb
**Purpose**: Alternative feature engineering approach with correlation analysis

**Key Activities**:
- Uses mean imputation as simpler alternative to Lasso regression
- Aggregates usage patterns over time windows (3, 6, 12 months)
- Creates trend features and usage growth indicators
- Pivots customer interaction data into feature format
- Engineers temporal features and customer lifecycle indicators
- Performs extensive correlation analysis between features
- Implements network-based feature selection to remove highly correlated variables

**Key Insights**: Simpler imputation methods.

### 2.baseline-models.ipynb
**Purpose**: Initial model evaluation with default parameters

**Key Activities**:
- Tests six different algorithms: SVM, Random Forest, Histogram Gradient Boosting, XGBoost, LightGBM, and CatBoost
- Evaluates models using cross-validation with stratified sampling
- Establishes performance baselines before hyperparameter optimization
- Identifies class imbalance challenges affecting all models
- Documents initial F1-scores and recall metrics

**Key Insights**: All baseline models struggle with the imbalanced dataset, achieving low recall scores, highlighting the need for optimization and class balancing techniques.

### 3a.tuning-models.ipynb
**Purpose**: Hyperparameter optimization using Optuna (initial approach)

**Key Activities**:
- Implements systematic hyperparameter tuning for all six models
- Optimizes for F1-score as primary objective metric
- Applies cross-validation during optimization to prevent overfitting
- Saves optimized model configurations for subsequent evaluation

**Key Insights**: Proper hyperparameter tuning significantly improves model performance, particularly for tree-based models on imbalanced data.

### 3b.tuning-models-optimal_f1.ipynb
**Purpose**: Enhanced hyperparameter optimization with optimal F1 threshold finding

**Key Activities**:
- Uses Optuna framework with 150 trials per model
- Optimizes models using dynamic threshold selection rather than fixed 0.5 threshold
- Tracks execution times and optimal thresholds for each trial
- Tests all six models: XGBoost, Random Forest, HistGradientBoosting, LightGBM, and CatBoost

**Key Insights**: Random Forest achieves best F1-score (0.524) with optimal threshold tuning, significantly outperforming other models. XGBoost shows best computational efficiency at 0.53s per trial.

### 3c.tuning-models-fv1-optimalf1.ipynb
**Purpose**: Final hyperparameter optimization using feature version 1 with optimal F1 thresholds

**Key Activities**:
- Uses Optuna framework with 150 trials per model
- Uses alternative feature engineering approach (feature version 1) for optimization
- Implements 100 trials per model with optimal F1 threshold discovery
- Compares performance using different feature sets to validate feature engineering choices
- Evaluates impact of feature engineering approach on model performance

**Key Insights**: Random Forest maintains superior performance (0.514) even with alternative features, confirming its robustness. Results show that feature engineering approach has modest impact compared to model selection and hyperparameter tuning.


### 4a.evaluate-models.ipynb
**Purpose**: Comprehensive model evaluation and selection

**Key Activities**:
- Evaluates all optimized models on hold-out test set
- Compares performance across multiple metrics (F1-score, precision, recall, ROC-AUC)
- Analyzes feature importance for best-performing models
- Creates detailed performance comparison tables
- Documents final model selection rationale

**Key Insights**: Random Forest emerges as the top performer with the highest recall (0.42), making it ideal for identifying upselling opportunities where missing potential customers is costly.

### 5a.tuning-segments-models-age.ipynb
**Purpose**: Segmented model optimization based on customer age

**Key Activities**:
- Segments customers into two age groups: under 55 and 55+ years
- Optimizes separate models for each age segment using Optuna
- Tests hypothesis that age-specific models perform better than global models
- Implements 100 trials per model per segment with optimal F1 threshold discovery
- Evaluates XGBoost, Random Forest, LightGBM, CatBoost, and HistGradientBoosting for each segment

**Key Insights**: Age-based segmentation shows modest performance improvements. Younger customers (age < 55) achieve better F1-scores across all models, with HistGradientBoosting performing best at 0.164 F1-score.

### 5b.tuning-segments-models-contract_life.ipynb
**Purpose**: Segmented model optimization based on contract lifetime

**Key Activities**:
- Segments customers by contract lifetime: short-term (< 1000 days) vs long-term (≥ 1000 days)
- Optimizes separate models for each contract duration segment
- Uses same optimization framework as age segmentation with 100 trials per model
- Analyzes whether contract tenure affects model performance patterns
- Compares segmented vs. global modeling approaches

**Key Insights**: Contract lifetime segmentation yields similar results to age segmentation. CatBoost performs best for short-term contracts (0.173 F1-score) while XGBoost leads for long-term contracts (0.160 F1-score).

### 5c.tuning-segments-models-gb.ipynb
**Purpose**: Segmented model optimization based on data usage

**Key Activities**:
- Segments customers by available data: low usage (< 25GB) vs high usage (≥ 25GB)
- Optimizes models separately for each data usage pattern
- Tests whether data consumption behavior affects predictive model performance
- Implements comprehensive model evaluation across both segments
- Analyzes feature importance variations between usage segments

**Key Insights**: Data usage segmentation shows the most promising results among segmentation approaches. XGBoost achieves highest performance (0.185 F1-score) for low data users, while CatBoost performs best (0.148 F1-score) for high data users.

### 6a.evaluate-models-age.ipynb
**Purpose**: Evaluation of age-segmented models on hold-out test data

**Key Activities**:
- Loads optimized age-segmented models and evaluates on test data
- Compares performance between age groups (< 55 vs ≥ 55 years)
- Analyzes model agreement/disagreement patterns within each age segment
- Creates comprehensive evaluation reports with confusion matrices and performance metrics
- Saves trained models as artifacts for potential production use

**Key Insights**: Age-segmented models show varied performance across segments. Younger customers are easier to predict with Random Forest achieving 0.154 F1-score, while older customers prove more challenging with XGBoost leading at 0.141 F1-score.

### 6a.evaluate-models-contract_life.ipynb
**Purpose**: Evaluation of contract lifetime-segmented models on test data

**Key Activities**:
- Evaluates contract duration-based segmented models on hold-out test set
- Compares short-term vs long-term contract customer prediction performance
- Analyzes correlation patterns between models within each contract segment
- Documents model disagreement statistics and optimal threshold performance
- Generates final model artifacts for contract-based segmentation approach

**Key Insights**: Contract lifetime segmentation shows Random Forest excelling for short-term contracts (0.148 F1-score) while Random Forest also leads for long-term contracts (0.144 F1-score), suggesting consistent model superiority across contract durations.

### 6a.evaluate-models-gb.ipynb
**Purpose**: Evaluation of data usage-segmented models on test data  

**Key Activities**:
- Evaluates models segmented by data usage patterns on test data
- Compares low data usage (< 25GB) vs high data usage (≥ 25GB) customer predictions
- Analyzes model correlation and disagreement patterns between usage segments
- Creates detailed performance analysis including precision-recall curves
- Stores optimized segmented models for potential deployment

**Key Insights**: Data usage segmentation demonstrates strong differentiation in model performance. Low usage customers show higher predictability with better F1-scores across all models, while high usage customers require more sophisticated modeling approaches.

### 6b.tuning-segments-meta-learner.ipynb
**Purpose**: Meta-learning optimization for ensemble and stacking approaches

**Key Activities**:
- Implements stacking classifiers using Ridge regression as meta-learner
- Combines predictions from age, contract lifetime, and data usage segmented models
- Optimizes meta-learner hyperparameters using Optuna with 50 trials
- Tests various combinations of segmented models (age+days, age+data, days+data)
- Evaluates whether ensemble approaches outperform individual segmented models

**Key Insights**: Meta-learning approaches achieve limited improvement over individual models. The best meta-learner combining all three segmentation approaches achieves 0.273 F1-score, but this doesn't significantly exceed the performance of well-tuned individual models, suggesting diminishing returns from ensemble complexity.

### 7.evaluate-meta-learners.ipynb
**Purpose**: Meta-learning and ensemble method evaluation

**Key Activities**:
- Implements stacking classifiers combining multiple base models
- Tests segmented modeling approaches based on customer characteristics (age, contract lifetime, data usage)
- Evaluates whether ensemble methods can improve upon single model performance
- Compares meta-learning results against traditional approaches
- Analyzes computational complexity vs. performance trade-offs

**Key Insights**: Meta-learning approaches do not provide significant performance improvements over well-tuned individual models, making the simpler Random Forest approach preferable for production.

### 8.production.ipynb
**Purpose**: Final model training and production preparation

**Key Activities**:
- Trains Random Forest model on complete dataset using optimal hyperparameters
- Creates production artifacts including model serialization and feature importance rankings
- Implements feature preprocessing pipeline for deployment
- Generates model documentation and performance summaries
- Prepares model for integration into production systems

**Key Insights**: The production Random Forest model achieves strong recall performance while maintaining interpretability through feature importance analysis, making it suitable for business deployment.

## Final Model Selection

**Random Forest was selected as the production model** due to its superior recall performance (0.42), which is crucial for identifying potential upselling customers. Missing a customer who would upgrade (false negative) is more costly than targeting a customer who won't upgrade (false positive), making recall the primary selection criterion.

## Technical Stack

- **Data Processing**: Polars
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna for hyperparameter tuning
- **Evaluation**: Cross-validation with stratified sampling for imbalanced data
- **Visualization**: Plotyl

## Business Impact

The final model enables Magenta to identify approximately 42% of potential upselling customers, significantly improving campaign efficiency compared to random targeting. This targeted approach reduces marketing costs while maximizing revenue generation from existing customers.