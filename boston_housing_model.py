import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load


class HousingModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor()
        self.pipeline = None
        self.train_data = None
        self.test_data = None
        self.labels = None

    def load_data(self):
        """Load the dataset from the given path."""
        housing_data = pd.read_csv(self.data_path)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in split.split(housing_data, housing_data["CHAS"]):
            self.train_data = housing_data.loc[train_idx]
            self.test_data = housing_data.loc[test_idx]

    def preprocess_data(self):
        """Create a pipeline for data preprocessing and transform the training set."""
        self.labels = self.train_data["MEDV"].copy()
        features = self.train_data.drop("MEDV", axis=1)

        # Create a pipeline for imputation and scaling
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ])

        # Fit the pipeline on the training features and transform them
        features_prepared = self.pipeline.fit_transform(features)
        return features_prepared

    def train_model(self, features_prepared):
        """Train the model using the preprocessed features."""
        self.model.fit(features_prepared, self.labels)

    def evaluate_model(self, features_prepared):
        """Evaluate the model using cross-validation and return the RMSE."""
        scores = cross_val_score(self.model, features_prepared, self.labels,
                                 scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        return rmse_scores

    def save_model(self, filename='housing_model.joblib'):
        """Save the trained model to a file."""
        dump(self.model, filename)

    def test_model(self):
        """Test the model on the test set and calculate the RMSE."""
        X_test = self.test_data.drop("MEDV", axis=1)
        Y_test = self.test_data["MEDV"].copy()
        X_test_prepared = self.pipeline.transform(X_test)

        final_predictions = self.model.predict(X_test_prepared)
        final_mse = mean_squared_error(Y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        return final_rmse

    def predict(self, features):
        """Predict housing prices for new data points."""
        # Ensure the features have the correct column names
        features_df = pd.DataFrame(features, columns=self.train_data.drop("MEDV", axis=1).columns)
        features_prepared = self.pipeline.transform(features_df)
        return self.model.predict(features_prepared)


def run_housing_project():
    # Initialize model with data path
    model = HousingModel(data_path='data/data.csv')

    # Load and preprocess data
    model.load_data()
    preprocessed_features = model.preprocess_data()

    # Train the model
    model.train_model(preprocessed_features)

    # Evaluate the model
    evaluation_scores = model.evaluate_model(preprocessed_features)
    print(f"Cross-Validation RMSE Scores: {evaluation_scores}")
    print(f"Mean RMSE: {evaluation_scores.mean()}")
    print(f"Standard Deviation of RMSE: {evaluation_scores.std()}")

    # Save the trained model
    model.save_model()

    # Test the model on test set
    test_rmse = model.test_model()
    print(f"RMSE on Test Set: {test_rmse}")

    # Example prediction
    sample_features = np.array([[-0.43942006, 3.12628155, -1.12165014, -0.27288841,
                                 -1.42262747, -22.24141041, -6.31238772, 2.61111401,
                                 -7.0016859, -0.5778192, -0.97491834, 0.41164221,
                                 -0.86091034]])
    prediction = model.predict(sample_features)
    print(f"Prediction for Sample Data: {prediction}")


if __name__ == "__main__":
    run_housing_project()
