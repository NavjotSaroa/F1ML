"""
Author: Navjot Saroa

Step 3 of process:
    At this point, the data has been collected and organised, with only the needed features making it to this point.
    Over here we can test machine learning models or have a trained model ready to go.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

class MLTime():
    def __init__(self, f, model):
        """Constructor"""
        self.model = model
        self.f = f          # f is the file name

    def get_df(self):
        """
        Returns a dataframe read from parquet file

        Sends data to new_col_m_currentLapTimeInMS_next
        """
        print("Creating dataframe...")
        df = pd.read_parquet(f'{self.f}', engine='pyarrow')
        print("Done!")
        return df

    def new_col_m_currentLapTimeInMS_next(self):
        """
        Creates a new column called m_currentLapTimeInMS_next.
        Copies all the data in the m_currentLapTimeInMS column to the new one and shifts it to the previous lap.
        Now every row in the m_currentLapTimeInMS_next is lined up with the timing in the next lap, allowing for a prediction.

        Receives data from get_df
        Sends data to generate_train_test_data and create_model
        """
        df_loaded = self.get_df()                                   # Load the DataFrame from a Parquet file
        print("Adjusting dataframe...")
        df_loaded["lapTimePointToPointInMS"] = df_loaded["m_currentLapTimeInMS"].shift(-3900)

        df_loaded['m_lastLapTimeInMS'] = df_loaded['m_lastLapTimeInMS'].shift(-3900)

        df_loaded['lapTimePointToPointInMS'] += df_loaded['m_lastLapTimeInMS'] - df_loaded['m_currentLapTimeInMS']

        df_loaded = df_loaded.drop("session_UID", axis = 1)
        print("Done!")
        return df_loaded.iloc[:-3900]

    def generate_train_test_data(self, test_size):
        """
        Creates the train/test split

        Receives data from new_col_m_currentLapTimeInMS_next
        Sends data to model_test
        """
        df = self.new_col_m_currentLapTimeInMS_next()
        print("Generating train/test split...")
        X = df.iloc[:, 3:-1]                                        # Use all features except m_currentLapTimeInMS (1), m_currentLapTimeInMS_next (-1, this is the target), session_UID (-2), lap (-3)
        print(X)
        y = df['lapTimePointToPointInMS']
        split = train_test_split(X, y, test_size = test_size, random_state = 42)  # Split the data into training and testing sets (80% train, 20% test)
        print("Done!")
        return split
        
    def model_test(self, margin_of_error, ratio = 0.2):
        """
        Tests the model for the given margin of error with the chosen ratio of train/test split

        Receives data from generate_train_test_data
        """
        X_train, X_test, y_train, y_test = self.generate_train_test_data(ratio)
        print("Running test...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.find_importance(self.model, self.new_col_m_currentLapTimeInMS_next().iloc[:,1:-3])

        errors_list = (y_pred - y_test).to_list()                   # Calculate the differences between predictions and true values
        absolute_errors = abs(y_pred - y_test)                      # Calculate the absolute differences between predictions and true values
        self.error_distribution_grapher(errors_list)

        within_50ms = sum(absolute_errors <= margin_of_error)       # Calculate the success rate within 50ms of the actual values
        success_rate = (within_50ms / len(y_test)) * 100
        print("Done!")
        print(success_rate)
        return success_rate

    def CIFinder(self, X_test, y_test):
        """
        Finds the confidence interval for all predictions, practically useless in the case of RandomForestRegressor,
        or the DecisionTreeRegressor but keeping it here for when I come back to tinker with TensorFlow.

        Helper to model_test
        """
        CF_bounds = []
        y_pred = self.model.predict(X_test)
        errors_list = (y_pred - y_test).to_list()
        for element in range(len(X_test)):
            if abs(errors_list[element]) > 2000:
                sample_list = sorted([self.model.predict(X_test.iloc[[element]])[0] for i in range(100)])
                mean =  np.mean(sample_list)
                sd = np.std(sample_list)
                rng = (mean - sd, mean + sd)
                CF_bounds.append(rng)
        return CF_bounds
    
    def error_distribution_grapher(self, errors_list):
        """
        Says it on the tin, graphs the frequency of the errors, can use if I feel like it.

        Helper to model_test
        """
        mean_error = np.mean(errors_list)
        std_error = np.std(errors_list)

        # Generate a range of x values for the bell curve
        x_range = np.linspace(min(errors_list), max(errors_list), 1000)

        # Generate the probability density function (PDF) for the normal distribution
        pdf = stats.norm.pdf(x_range, mean_error, std_error)

        # Plot the normal distribution
        plt.plot(x_range, pdf, label="Normal Distribution of Errors")
        plt.title("Error Distribution (Bell Curve)")
        plt.xlabel("Error (seconds)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def create_model(self):
        """
        Creates a model using the entire data set

        Receives data from new_col_m_currentLapTimeInMS_next
        """
        df = self.new_col_m_currentLapTimeInMS_next()
        print("Generating pickle file...")
        X = df.iloc[:, 3:-1]  
        y = df['lapTimePointToPointInMS']
        self.model.fit(X, y)
        print("Done!")
        return self.model


    def make_pickle_file(self):
        """
        Just makes the pickle file for future use.

        Helper to create_model
        """
        with open('models/lapTime_trained_model.pkl', 'wb') as file:
            pickle.dump(self.create_model(), file)

    def find_importance(self, model, df):
        """
        Plots a bar graph of all the features and quantifies how important each one is.

        Helper to model_test
        """
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        print(df.columns, importances)
        feature_names = [f"feature {i}: {name}" for i, name in enumerate(df.columns)]

        forest_importances = pd.Series(importances, index=feature_names)
        fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the figure

        # Plot the feature importances
        forest_importances.plot.bar(yerr=std, ax=ax)

        # Set the title and labels
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")

        # Customize x-axis labels: rotate them 45 degrees and reduce the font size
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        # Display the plot
        plt.tight_layout()  # This ensures everything fits well
        plt.show()

if __name__ == "__main__":
    model = RandomForestRegressor()
    f = "parquet_files/lap_data.parquet"
    MLTime(f, model).model_test(50)
