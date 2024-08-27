"""
Author: Navjot Saroa

Step 5 of process:
    This is the last step for all the prediction work, here, we will use a generational learning system
    We produce initial nodes, an equal distribution for soft, medium, and hard tyres. We have each node
    simulate a full race on its own, keeping track of various bits of information. After the simulation
    we choose the best performing nodes to reproduce children, who all have a 1% chance of mutating, who 
    repeat the process. After the final generation we pick the best performing node and that node will
    represent the best pit strategy and the predicted race time.
"""

import pandas as pd
import numpy as np
from FeaturePredictor import FeaturePredictor
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp    
import time
import random
from math import log, e
import warnings
import multiprocessing as mp
import gc

def process_actor(actor, race_length = 36, fp = FeaturePredictor()):
    """
    Not kept within any class so that multiprocessing can be done without any errors. This function
    simulates an entire race for a single node.
    """
    # The model works fine so not sure why it yells this at me, either way this should quieten it down
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    features_models, lap_time_model = fp.load_models()

    for i in range(1, race_length):
        print("Lap: ", i)
        if not actor.parent_features:
            lap_time = fp.predict_lap_time(lap_time_model, actor.features_lst[-1])              # Predicts lap time
            actor.total_lap_time += lap_time
            predicted_features = fp.predict_features(features_models, actor.features_lst[-1])   # Predicts features
            tyre_age = float(predicted_features["m_tyresAgeLaps"].iloc[0])
            if actor.trigger_pit_stop(tyre_age):                                                # Triggers pit stop depending on tyre age
                predicted_features["m_tyresAgeLaps"] = 1.0                                      # Reset lap info
                predicted_features["lap"] = 2.0                                        
                tyre_status = predicted_features[["isSoft", "isMed", "isHard"]].values[0]       # Shows which tyre is currently being used
                current_tyre = sum([tyre * 2**index for index, tyre in enumerate(tyre_status)])
                change_to = actor.change_tyre(int(current_tyre))
                if change_to == 1:
                    predicted_features.loc[:, ["isSoft", "isMed", "isHard"]] = [1.00, 0.00, 0.00]
                elif change_to == 2:
                    predicted_features.loc[:, ["isSoft", "isMed", "isHard"]] = [0.00, 1.00, 0.00]
                elif change_to == 4:
                    predicted_features.loc[:, ["isSoft", "isMed", "isHard"]] = [0.00, 0.00, 1.00]
                actor.total_lap_time += 20.4                                                    # Add time penalty for pitting
                actor.pit_stops.append((i, change_to))

            else:
                actor.pit_stops.append((i, 0))
            actor.lap_time_lst.append(actor.total_lap_time - actor.lap_time_lst[-1])
            actor.features_lst.append(predicted_features)
            gc.collect()

        else:                                                                                   # Newer generations after 1st start here
            if random.random() >= 0.01:                                                         # Mutation probabilty
                actor.features_lst.append(actor.parent_features[i])

            else:                                                                               # Clean up if node mutates
                actor.parent_features = None
                actor.pit_stops = actor.pit_stops[:i]
                actor.lap_time_lst = actor.lap_time_lst[:i]
                actor.total_lap_time = sum(actor.lap_time_lst)


    return actor

def convert_milliseconds(ms):
    """
    Convert milliseconds to hours, minutes, seconds, and remaining milliseconds
    """
    hours = ms // (1000 * 60 * 60)
    ms = ms % (1000 * 60 * 60)
    
    minutes = ms // (1000 * 60)
    ms = ms % (1000 * 60)
    
    seconds = ms // 1000
    milliseconds = ms % 1000
    
    # Format the result as "hours:minutes:seconds.milliseconds"
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"



class StrategyNode:
    def __init__(self, initial_feature, initial_tyre):
        """Constructor"""
        self.parent_features = None
        self.initial_feature = initial_feature
        self.features_lst = [initial_feature]
        self.lap_time_lst = [0]
        self.total_lap_time = 0
        self.pit_stops = [(0, initial_tyre)]

    def trigger_pit_stop(self, lap, a = 0.31915, b = -38.30387):       
        """
        Determines whether a pitstop should occur by random chance, increasing
        the probability of pitting as tyres get older. Can change function
        to better represent tyre wear.
        """
        x = random.random()
        probability_func = log(1 + (e ** (a * (lap + b))))

        if x <= probability_func:
            return True
        else:
            return False
        
    def change_tyre(self, current_tyre):
        """
        Tyre compounds are 1-hot encoded, this function takes advantage
        of that, determining which tyre to change to on a 50/50 chance
        and just representing the tyre as a power of 2. This means we 
        only need to shift the value left or right by 1.
        """
        choice = random.random()
        if choice >= 0.5:
            current_tyre = current_tyre >> 1
            if current_tyre < 1:
                return 1
        else:
            current_tyre = current_tyre << 1
            if current_tyre > 4:
                return 4
        return current_tyre

class EvaluateStrategy:
    def __init__(self, models, seeds, population = 10):
        self.actors = [StrategyNode(seeds[0], 1) for _ in range(population)]
        self.actors += [StrategyNode(seeds[1], 2) for _ in range(population)]
        self.actors += [StrategyNode(seeds[2], 4) for _ in range(population)]

        self.feature_models, self.lap_time_model = models

    def find_strategy(self, pool_size = 6, generations = 5):
        """
        Processes the race strategy for all the nodes in self.actors, generates children
        nodes from the best performing ones, and repeats process until the generation limit
        has been reached
        """
        while generations != 0:
            print("Generations left: ", depth)
            actors_lst = []
            with mp.Pool(processes=pool_size) as pool:
                # Process actors in parallel and collect the results
                actors_lst = pool.map(process_actor, self.actors)
            
            # Sort the processed actors by the sum of lap_time_lst
            actors_lst.sort(key=lambda actor: actor.total_lap_time)
            
            actors_lst = [actor for actor in actors_lst if self.valid_strategy(actor.pit_stops)]
            # Create new generation
            if len(actors_lst) >= 6:
                best_performers = actors_lst[:6]

            self.actors = []
            for parent in best_performers:
                for _ in range(5):  # Transfer parent attributes to children as necessary
                    child = StrategyNode(parent.initial_feature, parent.pit_stops[0][1])
                    child.parent_features = parent.features_lst
                    child.pit_stops = parent.pit_stops
                    child.lap_time_lst = parent.lap_time_lst
                    child.total_lap_time = parent.total_lap_time
                    self.actors.append(child)

            generations -= 1


        print("Pit Strategy: ", [pit for pit in actors_lst[0].pit_stops if pit[1] != 0])
        print("Predicted Race Time: ", convert_milliseconds(actors_lst[0].total_lap_time))
    
    def valid_strategy(self, pit_lst):
        """
        A strategy is only valid if at least 2 different tyre compounds have been
        used in the race, this function checks for it.
        """
        pit_set = set([pit[1] for pit in pit_lst if pit[1] != 0])
        if len(pit_set) > 1:
            return True
        else:
            return False


if __name__ == "__main__":
    model = RandomForestRegressor()
    f = "parquet_files/lap_data.parquet"
    fp = FeaturePredictor(f, model)
    df = fp.get_df()
    df = df.dropna()
    df = df.iloc[:-3900,3:]

    # Create seeds
    first_rows = df.groupby('session_UID').first().reset_index()
    hard_seed = first_rows[first_rows['session_UID'] == "6381938460888499015"].drop("session_UID", axis = 1)
    med_seed = first_rows[first_rows['session_UID'] == "358259683733558242"].drop("session_UID", axis = 1)
    soft_seed = first_rows[first_rows['session_UID'] == "6752990771969046537"].drop("session_UID", axis = 1)

    seeds = [soft_seed, med_seed, hard_seed]


    models = fp.load_models()
    EvaluateStrategy(models, seeds).find_strategy()
