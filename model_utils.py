import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class HierarchicalPasswordClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements the Hierarchical Classification Model (BoW + LogReg) 
    as described in the paper 'Classifying user-created passwords...'.
    
    Structure:
    - Stage 1: Binary classification (Class 0 vs. Class 1 & 2).
    - Stage 2: Multi-class classification (Class 1 vs. Class 2) for survivors of Stage 1.
    - Shared Vocabulary: Both stages use the same BoW feature space.
    """
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char')
        self.model_stage1 = LogisticRegression(max_iter=5000)
        self.model_stage2 = LogisticRegression(max_iter=5000)

    def fit(self, X_raw, y):
        """
        Train the hierarchical model.
        X_raw: List of password strings.
        y: List of integer labels (0: Weak, 1: Normal, 2: Strong).
        """
        # Feature Extraction (Bag-of-Words) - Shared Vocabulary
        X = self.vectorizer.fit_transform(X_raw)
        
        # --- Stage 1: Class 0 vs. Others (1, 2) ---
        # Label 0 remains 0; Labels 1 and 2 become 1 (temporary binary label)
        y_stage1 = (y != 0).astype(int)
        self.model_stage1.fit(X, y_stage1)
        
        # --- Stage 2: Class 1 vs. Class 2 ---
        # Filter data: Only train on samples that are NOT weak (Label != 0)
        mask_stage2 = (y_stage1 == 1)
        X_stage2 = X[mask_stage2]
        y_stage2 = y[mask_stage2]
        
        # Train Stage 2 only if there are samples for both classes
        if len(np.unique(y_stage2)) > 1:
            self.model_stage2.fit(X_stage2, y_stage2)
        else:
            # Fallback if training set lacks minority classes (rare case)
            print("Warning: Insufficient data for Stage 2 training.")
            self.model_stage2 = None
            
        return self

    def predict(self, X_raw):
        """
        Predict labels for new passwords.
        """
        # Transform input using the learned vocabulary
        X = self.vectorizer.transform(X_raw)
        
        # Stage 1 Prediction
        y_pred_stage1 = self.model_stage1.predict(X)
        
        # Initialize final predictions with Stage 1 results
        # If Stage 1 predicts 0, final is 0. If 1, it's a candidate for Stage 2.
        y_final = y_pred_stage1.copy()
        
        # Identify indices where Stage 1 predicted "Not Weak"
        stage2_indices = np.where(y_pred_stage1 == 1)[0]
        
        if len(stage2_indices) > 0 and self.model_stage2 is not None:
            # Extract features for these candidates
            X_stage2 = X[stage2_indices]
            # Stage 2 Prediction (returns original labels 1 or 2)
            y_pred_stage2 = self.model_stage2.predict(X_stage2)
            # Update final predictions
            y_final[stage2_indices] = y_pred_stage2
            
        return y_final