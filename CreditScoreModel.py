import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CreditScoringModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}

    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic credit data for demonstration
        """
        np.random.seed(42)

        # Generate features
        data = {
            'age': np.random.normal(40, 12, n_samples).clip(18, 80),
            'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000),
            'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 0.8,
            'credit_history_length': np.random.exponential(8, n_samples).clip(0, 30),
            'num_credit_accounts': np.random.poisson(4, n_samples).clip(0, 20),
            'payment_history_score': np.random.beta(8, 2, n_samples) * 100,
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'num_late_payments': np.random.poisson(2, n_samples).clip(0, 20),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
            'savings_account_balance': np.random.lognormal(8, 1.5, n_samples).clip(0, 100000)
        }

        # Add categorical features
        data['employment_type'] = np.random.choice(['full_time', 'part_time', 'self_employed', 'unemployed'],
                                                 n_samples, p=[0.6, 0.2, 0.15, 0.05])
        data['home_ownership'] = np.random.choice(['own', 'rent', 'mortgage'],
                                                n_samples, p=[0.3, 0.4, 0.3])
        data['education_level'] = np.random.choice(['high_school', 'bachelor', 'master', 'phd'],
                                                 n_samples, p=[0.3, 0.4, 0.25, 0.05])

        df = pd.DataFrame(data)

        # Create target variable based on realistic credit scoring logic
        credit_score = (
            df['income'] / 1000 * 0.3 +
            (100 - df['debt_to_income_ratio'] * 100) * 0.25 +
            df['payment_history_score'] * 0.2 +
            df['credit_history_length'] * 2 +
            (100 - df['credit_utilization'] * 100) * 0.15 +
            np.maximum(0, 10 - df['num_late_payments']) * 3 +
            df['employment_length'] * 0.5 +
            df['savings_account_balance'] / 1000 * 0.1 +
            np.random.normal(0, 10, n_samples)  # Add some noise
        )

        # Convert to binary classification (good/bad credit)
        df['creditworthy'] = (credit_score > np.percentile(credit_score, 30)).astype(int)

        return df

    def preprocess_data(self, df):
        """
        Preprocess the data: handle missing values, encode categorical variables, scale features
        """
        df_processed = df.copy()

        # Handle missing values (if any)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns

        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])

        categorical_imputer = SimpleImputer(strategy='most_frequent')
        if len(categorical_cols) > 0:
            df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])

        # Encode categorical variables
        for col in categorical_cols:
            if col != 'creditworthy':  # Don't encode target variable
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                df_processed.drop(col, axis=1, inplace=True)

        return df_processed

    def feature_engineering(self, df):
        """
        Create additional features that might be predictive
        """
        df_engineered = df.copy()

        # Create interaction features
        df_engineered['income_to_age_ratio'] = df_engineered['income'] / df_engineered['age']
        df_engineered['debt_per_account'] = (df_engineered['debt_to_income_ratio'] * df_engineered['income']) / np.maximum(1, df_engineered['num_credit_accounts'])
        df_engineered['savings_to_income_ratio'] = df_engineered['savings_account_balance'] / df_engineered['income']

        # Create risk indicators
        df_engineered['high_utilization'] = (df_engineered['credit_utilization'] > 0.7).astype(int)
        df_engineered['frequent_late_payments'] = (df_engineered['num_late_payments'] > 5).astype(int)
        df_engineered['low_payment_history'] = (df_engineered['payment_history_score'] < 60).astype(int)

        # Binned features
        df_engineered['age_group'] = pd.cut(df_engineered['age'],
                                          bins=[0, 25, 35, 50, 65, 100],
                                          labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly'])
        df_engineered['income_bracket'] = pd.cut(df_engineered['income'],
                                               bins=[0, 30000, 50000, 80000, 200000],
                                               labels=['low', 'medium', 'high', 'very_high'])

        # Encode new categorical features
        categorical_features = ['age_group', 'income_bracket']
        for col in categorical_features:
            le = LabelEncoder()
            df_engineered[col + '_encoded'] = le.fit_transform(df_engineered[col])
            self.label_encoders[col] = le
            df_engineered.drop(col, axis=1, inplace=True)

        return df_engineered

    def train_models(self, X, y):
        """
        Train multiple classification models
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }

        # Train and evaluate models
        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Use scaled data for Logistic Regression, original data for tree-based models
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_

        self.models = results
        self.X_test = X_test
        self.X_train = X_train
        self.feature_names = X.columns.tolist()

        return results

    def print_model_comparison(self):
        """
        Print comparison of model performances
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)

        comparison_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'Accuracy': [results['accuracy'] for results in self.models.values()],
            'Precision': [results['precision'] for results in self.models.values()],
            'Recall': [results['recall'] for results in self.models.values()],
            'F1-Score': [results['f1_score'] for results in self.models.values()],
            'ROC-AUC': [results['roc_auc'] for results in self.models.values()]
        })

        print(comparison_df.round(4).to_string(index=False))

        # Find best model
        best_model = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
        print(f"\nBest performing model: {best_model['Model']} (ROC-AUC: {best_model['ROC-AUC']:.4f})")

    def plot_results(self):
        """
        Create visualizations for model evaluation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC Curves
        ax1 = axes[0, 0]
        for name, results in self.models.items():
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})")
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curves
        ax2 = axes[0, 1]
        for name, results in self.models.items():
            precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
            ax2.plot(recall, precision, label=f"{name}")
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Model Comparison Bar Chart
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        x = np.arange(len(metrics))
        width = 0.25

        for i, (name, results) in enumerate(self.models.items()):
            values = [results[metric] for metric in metrics]
            ax3.bar(x + i*width, values, width, label=name, alpha=0.8)

        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Feature Importance (Random Forest)
        ax4 = axes[1, 1]
        if 'Random Forest' in self.feature_importance:
            importance = self.feature_importance['Random Forest']
            indices = np.argsort(importance)[-10:]  # Top 10 features
            ax4.barh(range(len(indices)), importance[indices])
            ax4.set_yticks(range(len(indices)))
            ax4.set_yticklabels([self.feature_names[i] for i in indices])
            ax4.set_xlabel('Feature Importance')
            ax4.set_title('Top 10 Important Features (Random Forest)')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_creditworthiness(self, model_name='Random Forest', sample_data=None):
        """
        Make predictions for new data
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None

        model = self.models[model_name]['model']

        if sample_data is None:
            # Use a sample from test data
            sample_idx = np.random.choice(len(self.X_test))
            sample_data = self.X_test.iloc[sample_idx:sample_idx+1]

        # Make prediction
        if model_name == 'Logistic Regression':
            sample_scaled = self.scaler.transform(sample_data)
            prediction = model.predict(sample_scaled)[0]
            probability = model.predict_proba(sample_scaled)[0]
        else:
            prediction = model.predict(sample_data)[0]
            probability = model.predict_proba(sample_data)[0]

        print(f"\nCredit Assessment Results using {model_name}:")
        print("-" * 50)
        print(f"Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
        print(f"Probability of being creditworthy: {probability[1]:.3f}")
        print(f"Probability of default risk: {probability[0]:.3f}")

        return prediction, probability

# Main execution
def main():
    print("Credit Scoring Model - ML Implementation")
    print("=" * 50)

    # Initialize the model
    credit_model = CreditScoringModel()

    # Generate synthetic data
    print("Generating synthetic credit data...")
    df = credit_model.generate_synthetic_data(n_samples=5000)

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['creditworthy'].value_counts(normalize=True)}")

    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = credit_model.preprocess_data(df)

    # Feature engineering
    print("Engineering features...")
    df_final = credit_model.feature_engineering(df_processed)

    # Prepare features and target
    X = df_final.drop('creditworthy', axis=1)
    y = df_final['creditworthy']

    print(f"Final feature set: {X.shape[1]} features")

    # Train models
    print("\nTraining classification models...")
    results = credit_model.train_models(X, y)

    # Print results
    credit_model.print_model_comparison()

    # Plot results
    credit_model.plot_results()

    # Make sample predictions
    print("\nMaking sample predictions...")
    for _ in range(3):
        credit_model.predict_creditworthiness()
        print()

if __name__ == "__main__":
    main()