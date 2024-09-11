# Boston Housing Price Prediction

This project predicts housing prices in Boston using a RandomForestRegressor model. The data is preprocessed using a pipeline consisting of imputation and scaling. The model is trained and evaluated using cross-validation, and the results are tested on a separate test set.

## Project Structure


## Getting Started

### 1. Clone the repository
First, clone the repository to your local machine:

bash

git clone https://github.com/samarthpatel03/BostonHousingPrediction.git

cd BostonHousingPrediction

### 2. Install the dependencies
It is recommended to use a virtual environment. Install the required packages using the requirements.txt file:

pip install -r requirements.txt

### 3. Run the project
Ensure the dataset (data.csv) is in the data/ folder, then run the main Python script:

python boston_housing_model.py

### 4. Expected output
You will see the following outputs:

Cross-Validation RMSE Scores

RMSE on the Test Set

Predictions for sample data



#### Example output:

##### Cross-Validation RMSE Scores: [2.341 2.101 ...]

##### Mean RMSE: 2.22

##### Standard Deviation of RMSE: 0.12

##### RMSE on Test Set: 2.12

##### Prediction for Sample Data: [24.5]




