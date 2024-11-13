# 🚀 Hybrid Trading Account Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)]()
[![LLM](https://img.shields.io/badge/LLM-Llama%202-purple)]()

> A sophisticated trading account classification system combining traditional machine learning with state-of-the-art Large Language Models (LLMs) for high-accuracy pattern detection and risk assessment.

## 🌟 Key Features

- **Dual Classification System**: Innovative combination of ensemble ML and LLM-based approaches
- **State-of-the-Art Performance**: Achieved 100% accuracy on test set with LLM classifier
- **Advanced Feature Engineering**: Sophisticated ratio-based features for trading metrics
- **Robust Cross-Validation**: K-fold validation demonstrating consistent performance (Mean CV: 0.941)
- **Comprehensive Evaluation**: Extensive metrics including ROC-AUC, MCC, and Cohen's Kappa
- **Interactive Visualizations**: Feature importance and distribution analysis
- **Production-Ready Code**: Modular design with comprehensive documentation

## 📊 System Architecture

### Traditional ML Pipeline
```
Data → Feature Engineering → Ensemble Model (RF/XGB/SVM) → Classification
```

### LLM-Based Pipeline
```
Data → Feature Engineering → Embeddings → Nearest Neighbors → LLM Analysis → Classification
```

## 🔥 Performance Highlights

| Metric | Ensemble Model | LLM Model |
|--------|---------------|-----------|
| Accuracy | 0.952 | 1.000 |
| F1-Score | 0.957 | 1.000 |
| ROC-AUC | 0.982 | 1.000 |
| MCC | 0.901 | 1.000 |

## 🛠 Technical Implementation

### Traditional Ensemble Classifier
- Voting Classifier combining:
  - Random Forest
  - XGBoost
  - SVM
  - Logistic Regression
  - SGD Classifier

### LLM-Based Classifier
- Advanced embedding generation using BAAI/bge-m3
- Sophisticated nearest neighbor search
- Context-aware LLM analysis using Llama 2
- Custom prompt engineering for financial analysis

## 📈 Feature Engineering

Innovative financial ratios created:
- Profit-to-Deposit Ratio
- Commission-to-Profit Ratio
- Equity-to-Balance Ratio
- Win-Loss Ratio
- Average Profit per Trade

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
PyTorch
Scikit-learn
Hugging Face Transformers
LangChain
```

### Installation
```bash
git clone https://github.com/yourusername/trading-classification.git
cd trading-classification
pip install -r requirements.txt
```

### Quick Start
```python
from trading_classification import EnsembleClassifier, LLMClassifier

# Initialize classifiers
ensemble_clf = EnsembleClassifier()
llm_clf = LLMClassifier()

# Train models
ensemble_clf.fit(X_train, y_train)
llm_clf.fit(X_train, y_train)

# Make predictions
ensemble_preds = ensemble_clf.predict(X_test)
llm_preds = llm_clf.predict(X_test)
```

## 📊 Results Visualization

### Feature Importance
![Feature Importance](assets/feature_importance.png)

### Distribution Analysis
![Distribution Analysis](assets/distribution_analysis.png)

## 🧪 Model Validation

### Cross-Validation Results
- 5-fold CV scores: [1.0, 0.905, 1.0, 1.0, 0.8]
- Mean accuracy: 0.941
- Standard deviation: 0.0795

### Robustness Analysis
- Extensive testing across different market conditions
- Stability verification through bootstrap sampling
- Out-of-sample validation


----
# Model Architecture & Technical Details

## Ensemble Classifier Architecture

### Feature Processing Pipeline
```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### Model Components

1. **Random Forest**
   - Number of estimators: 295
   - Max depth: 15
   - Min samples leaf: 2
   - Class weight: balanced

2. **XGBoost**
   - Number of estimators: 166
   - Max depth: 4
   - Learning rate: 0.091
   - Subsample: 0.769
   - Colsample: 0.892

3. **Logistic Regression**
   - C: 3.334
   - Solver: saga
   - Max iterations: 10000

4. **SGD Classifier**
   - Alpha: 0.044
   - Loss: modified_huber
   - Penalty: l1

5. **SVM**
   - Kernel: sigmoid
   - C: 6.796
   - Probability estimates: True

## LLM-Based Classifier Architecture

### Embedding Generation
```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Nearest Neighbor Search
```python
nn = NearestNeighbors(
    n_neighbors=5,
    metric='cosine'
)
```

### Feature Engineering Components

1. **Ratio Calculations**
```python
def safe_divide(a, b, fill_value=0):
    try:
        return float(a) / (float(b) + 1e-5)
    except (ValueError, TypeError):
        return fill_value

def engineer_features(df):
    df['profit_to_deposit_ratio'] = df.apply(
        lambda row: safe_divide(row['profit'], row['deposits']), axis=1)
    df['commission_to_profit_ratio'] = df.apply(
        lambda row: safe_divide(row['commission'], row['profit']), axis=1)
    df['equity_to_balance_ratio'] = df.apply(
        lambda row: safe_divide(row['equity'], row['balance']), axis=1)
    df['win_loss_ratio'] = df.apply(
        lambda row: safe_divide(row['win_ratio'], 1 - row['win_ratio']), axis=1)
    df['avg_profit_per_trade'] = df.apply(
        lambda row: safe_divide(row['profit'], row['dealing_num']), axis=1)
    return df
```

### LLM Prompt Engineering

The system uses a carefully crafted prompt template for the LLM:

```python
prompt = f"""As an expert financial analyst, classify this trading account based on its performance metrics.

Context:
- Classes represent different trading performance levels
- Features describe trading behavior and outcomes
- Similar cases provided for reference

Key Information:
1. Query case features: {query_case}
2. Similar cases: {similar_cases}
3. Similar case classes: {similar_classes}
4. Class distribution: {class_distribution}
5. Feature importance: {feature_importance}

Instructions:
1. Analyze query case vs similar cases
2. Consider class distribution
3. Focus on important features
4. Utilize engineered features
5. Determine final classification

Classification:"""
```

## Model Evaluation

### Validation Strategy

1. **Train-Test Split**
   - Training: 82 samples (79.61%)
   - Testing: 21 samples (20.39%)

2. **Cross-Validation**
   - 5-fold stratified CV
   - Performance tracking across folds

3. **Metrics Tracked**
   - Accuracy
   - Precision
   - Recall 
   - F1-Score
   - ROC-AUC
   - Matthews Correlation
   - Cohen's Kappa

### Performance Analysis

1. **Feature Importance**
   - Top features identified
   - Correlation analysis
   - Distribution comparison

2. **Error Analysis**
   - Confusion matrix study
   - Misclassification patterns
   - Edge case handling

3. **Robustness Checks**
   - Out-of-sample testing
   - Cross-validation stability
   - Bootstrap validation

## Implementation Details

### Data Pipeline
```python
class DataPipeline:
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator()
    
    def process(self, data):
        validated_data = self.validator.validate(data)
        engineered_features = self.feature_engineering.transform(validated_data)
        processed_data = self.preprocessor.transform(engineered_features)
        return processed_data
```

### Model Pipeline
```python
class ModelPipeline:
    def __init__(self):
        self.ensemble_model = EnsembleClassifier()
        self.llm_model = LLMClassifier()
        self.evaluator = ModelEvaluator()
    
    def train_evaluate(self, X_train, y_train, X_test, y_test):
        # Train both models
        self.ensemble_model.fit(X_train, y_train)
        self.llm_model.fit(X_train, y_train)
        
        # Get predictions
        ensemble_preds = self.ensemble_model.predict(X_test)
        llm_preds = self.llm_model.predict(X_test)
        
        # Evaluate
        ensemble_metrics = self.evaluator.evaluate(y_test, ensemble_preds)
        llm_metrics = self.evaluator.evaluate(y_test, llm_preds)
        
        return ensemble_metrics, llm_metrics
```
----
## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Llama 2 team for the LLM infrastructure
- BAAI team for the embedding model
- Trading data providers for benchmark datasets

## 📬 Contact

- Your Name - [your.email@example.com]
- Project Link: [https://github.com/yourusername/trading-classification]
