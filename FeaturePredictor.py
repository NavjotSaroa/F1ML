"""
Author: Navjot Saroa

Step 4 of process:
    This file is to make predictions on the features of the model. The problem so far has been
    that no matter how good the lap time predictor will get, it can only predict one lap into the
    future, after that we have no clue what its features are so we cant predict the time either.

    Here we identify that these features are either deterministic, they change in consistent ways 
    or stay the same (like total distance increases by about 4320m a lap no matter what), or 
    non-deterministic, the stuff that is hard to predict. We use the deterministic features
    to predict the non-deterministic features. This new set of features is then used to 
    predict the lap times.

    The pipeline is very similar to MLTime.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from MLTime import MLTime
import pickle
import os
import warnings



class FeaturePredictor(MLTime):
    def __init__(self, f = "parquet_files/lap_data.parquet",
                model = RandomForestRegressor(), 
                deterministic_features = [
                    "m_totalDistance", 
                    "isSoft", "isMed", "isHard", 
                    "m_tyresAgeLaps"
                    ], 
                non_deterministic_features = [
                    "m_speed", "m_fuelInTank", 
                    "speedAtT1", "speedAtDRS1", 
                    "speedAtT3", "speedAtDRS2", 
                    "speedAtT10", "speedAtDRS3"
                    ]):

        super().__init__(f, model)
        self.deterministic_features = deterministic_features
        self.non_deterministic_features = non_deterministic_features
    
    def duplicate_and_transpose_features(self):
        """
        Some of the features need to be duplicated because they're going to be the targets.
        I also shift them by 3900 rows so that the duplicated columns now show data from the
        same point at the next lap.

        Same concept as in the training for the lap time model, this lets me pretend I have
        half a million laps of data instead of just 150-ish.
        """
        df = self.get_df().iloc[:-3900]
        print(df)
        print("Duplicating non-deterministic features...")
        for non_det_feature in self.non_deterministic_features:
            df[f"{non_det_feature}_target"] = df[non_det_feature].shift(-3900)

        print("Done!")
        return df.dropna()
    
    
    def generate_train_test_data(self, test_size):
        """
        Same job as in MLTime but for more features in one go
        """
        dataframes = self.create_targets_and_features()

        print("Generating train/test splits for all targets...")
        feature_df, targets = dataframes[0], dataframes[1]
        splits = {f"{target}_split":train_test_split(feature_df, targets[f"{target}"], test_size = test_size, random_state = 42) for target in targets}
        print("Done!")
        return splits

    def create_targets_and_features(self):
        """
        Helper to generate_train_test_data and to create_model_and_pickle_file
        """
        df = self.duplicate_and_transpose_features()

        print("Separating targets and features...")
        feature_df = df.iloc[:, 3:-8].drop("session_UID", axis = 1)
        target_df = df.iloc[:,-8:]
        targets_lst = target_df.columns.tolist()
        targets = {targets_lst[index]: target_df.iloc[:, index] for index in range(len(targets_lst))}
        print("Done!")
        return(feature_df, targets)


    def model_test(self, ratio = 0.05):
        """
        Same as in MLTime, runs tests, but this time I normalised the margin of errors to a percentage
        since we are dealing with various units.

        And here, a variety of margins of errors are tested.
        """
        splits = self.generate_train_test_data(ratio)
        print("Running tests...")

        for split in splits.keys():

            X_train, X_test, y_train, y_test = splits[split]
            print(f"Training for {split}")
            self.model.fit(X_train, y_train)
            print(f"{split} trained! Making predictions now...")
            y_pred = self.model.predict(X_test)

            errors_list = (y_pred - y_test).to_list() 
            for m_o_e in [0.0000005, 0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5, 5]:
                within_margin = sum(errors_list <= m_o_e * y_test) 
                success_rate = (within_margin / len(y_test)) * 100
                print(f"Success rate for feature {split} with m_o_e = {m_o_e}: {success_rate}%", )
        print("Testing Done!")

    def create_model_and_pickle_file(self):
        """
        Trains the model and then stores it in a models directory.
        """
        features, targets = self.create_targets_and_features()
        
        print("Creating pickle files...")
        for target in targets.keys():
            print(f"Fitting for {target}...")
            self.model.fit(features, targets[target])
            with open(f"models/{target[:-7]}_trained_model.pkl", 'wb') as file:
                pickle.dump(self.model, file)
            print(f"{target} fitted!")
        print("Done!")

    def iterative_predictive_testing(self):
        """
        Predicts features for a lap, then predicts the lap time for that lap using those features
        """
        models, lapTime_model = self.load_models()
        ref_df = self.get_df().dropna()
        test_df = self.get_df().dropna().iloc[:,3:]
        test_df = test_df.drop("session_UID", axis = 1)
        first_row = test_df.iloc[0]
        
        for i in range(40):
            print(f"Lap number: {i}")
            prediction_df = self.predict_features(models, first_row)

            first_row = prediction_df
            prediction_df_reshaped = prediction_df.iloc[0].values.reshape(1,-1)

            time_prediction = lapTime_model.predict(prediction_df_reshaped)
            print("Predicted time: ",time_prediction[0])

            percent_error = (-(time_prediction[0] - ref_df.iloc[3900*(i)]["m_lastLapTimeInMS"]) * 100 / ref_df.iloc[3900*(i)]["m_lastLapTimeInMS"])
            print(time_prediction[0] , ref_df.iloc[3900*(i)]["m_lastLapTimeInMS"], percent_error)

    def load_models(self):
        """
        Just loads up the models, it's time consuming so better just to do it all in one go
        """
        models = {}
        for filename in os.listdir("models"):
            # print(filename)
            if filename.endswith('.pkl'):  # Only process .pkl files
                model_name = os.path.splitext(filename)[0]  # Get the file name without the extension
                file_path = os.path.join("models", filename)
                if model_name == "lapTime_trained_model":
                    with open(file_path, 'rb') as file:
                        lapTime_model = pickle.load(file)
                        continue
                # Load the model and store it in the dictionary
                with open(file_path, 'rb') as file:
                    models[f"{model_name[:-14]}"] = pickle.load(file)
            
        return (models, lapTime_model)

    def predict_features(self, models, df_single_row):
        """
        models is the dictionary of models for each feature.
        df_single_row is the row of features you want to predict future features on.
        """
        init_df = self.get_df().dropna().iloc[:,3:]
        init_df = init_df.drop("session_UID", axis = 1)
        prediction_df_row = pd.DataFrame(np.nan, index=[0], columns=init_df.columns)
        prediction_df_row = prediction_df_row.set_index(df_single_row.index)
        for key in models.keys():                                               # Takes models from model directory
            df_single_row_reshaped = df_single_row.values.reshape(1, -1)    
            prediction = models[key].predict(df_single_row_reshaped)                # Makes prediction
            prediction_df_row[key] = prediction[0]
        prediction_df_row["m_totalDistance"] = df_single_row["m_totalDistance"] + 4324.311035   # Update deterministic features
        prediction_df_row["isSoft"] = df_single_row["isSoft"]
        prediction_df_row["isMed"] = df_single_row["isMed"]
        prediction_df_row["isHard"] = df_single_row["isHard"]
        prediction_df_row["m_tyresAgeLaps"] = df_single_row["m_tyresAgeLaps"] + 1
        prediction_df_row["lap"] = df_single_row["lap"] + 1
        

        return prediction_df_row
    
    def predict_lap_time(self, model, features):
        """
        Predicts lap time just once on the basis of the features given.
        model is the singular model made in MLTime.
        features is a row from the dataframe.
        """
        features_reshaped = features.values.reshape(1, -1)
        prediction = model.predict(features_reshaped)
        return prediction[0]

if __name__ == "__main__":
    # The model works fine so not sure why it yells this at me, either way this should quieten it down
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    
    model = RandomForestRegressor()
    f = "parquet_files/lap_data.parquet"
    predictor = FeaturePredictor(f, model)
    predictor.iterative_predictive_testing()

