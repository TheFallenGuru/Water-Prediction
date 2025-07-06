# Water Quality Prediction for Concrete Mixing

## Problem Statement
Water quality is critical for concrete mixing, affecting strength and durability. Poor quality leads to costly repairs. This project builds a machine learning model to predict water quality for concrete mixing using real-time measurements of pH, Chloride, Organic Carbon, Solids, Sulphate, and Turbidity.

### Objectives
- Use dataset with features: pH, Chloride, Organic Carbon, Solids, Sulphate, Turbidity.
- Develop ML models to classify water quality as "good" or "bad."
- Evaluate using Precision, Recall, Accuracy, F1-Score, Sensitivity, Specificity, and Confusion Matrix.

## Requirements
- **Dataset**: `Data.csv` with water characteristics and binary labels.
- **Models**: Logistic Regression, Random Forest, SVM, K-Nearest Neighbors.
- **Metrics**: Precision, Recall, Accuracy, F1-Score, Sensitivity, Specificity, Confusion Matrix.
- **Output**: Save best model, scaler, and visualizations.

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage
1. Clone repository:
   ```bash
   git clone https://github.com/your-username/water-quality-prediction.git
   cd water-quality-prediction
   ```
2. Add `Data.csv` to directory.
3. Run script:
   ```bash
   python water_quality_analysis.py
   ```
4. Outputs:
   - Console: Performance metrics.
   - Visualizations: `confusion_matrices.png`, `model_comparison.png`, `feature_importance.png`.
   - Saved: `scaler.pkl`, `best_water_quality_model.pkl`.

## Results
Model performance:
- **Logistic Regression**: Precision: 0.7377, Recall: 0.8782, Accuracy: 0.8485, F1-Score: 0.8018
- **Random Forest**: Precision: 0.6656, Recall: 0.6218, Accuracy: 0.7590, F1-Score: 0.6430
- **SVM**: Precision: 0.7377, Recall: 0.8782, Accuracy: 0.8485, F1-Score: 0.8018
- **KNN**: Precision: 0.7139, Recall: 0.7865, Accuracy: 0.8155, F1-Score: 0.7485

### Inferences
- Logistic Regression and SVM excel (F1-Score: 0.8018).
- Random Forest lowest (F1-Score: 0.6430) due to class imbalance.
- Feature importance emphasizes pH and Chloride.
- Best model (Logistic Regression) saved.

## Files
- `water_quality_analysis.py`: Main script.
- `Data.csv`: Input dataset (not included).
- `scaler.pkl`: Saved scaler.
- `best_water_quality_model.pkl`: Saved best model.
- `confusion_matrices.png`: Confusion matrices.
- `model_comparison.png`: Model metrics bar plot.
- `feature_importance.png`: Random Forest feature importance.

## Future Improvements
- Hyperparameter tuning.
- Additional features for improved accuracy.