from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, X, y):
        """Train the prediction model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate prediction metrics
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            return {
                'train_score': self.model.score(X_train, y_train),
                'test_score': self.model.score(X_test, y_test),
                'X_test': X_test,
                'y_test': y_test,
                'test_pred': test_pred
            }
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")

    def predict(self, X):
        """Make predictions with confidence intervals"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        try:
            predictions = self.model.predict(X)
            
            # Calculate simple confidence intervals (±2 standard deviations)
            std_dev = np.std(predictions)
            confidence_intervals = {
                'lower': predictions - 2 * std_dev,
                'upper': predictions + 2 * std_dev
            }
            
            return predictions, confidence_intervals
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
