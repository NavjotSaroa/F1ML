# F1 Machine Learning Race Strategy Generator

## TL;DR - Key Successes:
* **Implemented a robust data collection and processing system with zero data loss at 120Hz.**
* **Achieved a maximum accuracy of 99.975% with a precision window of 0.00001% for car telemetry data and a precision window of 0.07% for lap times, with limited data.**
* **Developed a custom race strategy model achieving 99.6% race time prediction accuracy**


This project is intended to showcase my learnings in the field of live data collection and processing, and machine learning within the context of F1 23 races. The goal was to build a system capable of collecting racing data in real-time, processing it, training machine learning models, and generating optimal pit-stop strategies for full-length races. 

## Project Phases:
The project was broken down into 3 phases:
1. Live data collection/processing
2. Model selection and training
3. Race strategy generation

Each phase has its own objectives and challenges, this README file discusses them.

## Phase 1: Live Data Collection/Processing:

 
Relevant files: UDP.py, structures.py, DBMakerL2.py

The F1 23 game transmits 14 different [packets of telemetry data](https://answers.ea.com/t5/General-Discussion/F1-23-UDP-Specification/m-p/12633159?attachment-id=704910) over a Wi-Fi network at up to 120Hz. This phase focused on collecting, filtering, and efficiently storing this data to create a dataset for machine learning model training.

Challenges included handling the high volume of data (over 100,000 packets per lap) while minimizing storage requirements. Using a "fingerprint" system to group similar packets and batch processing techniques, I eliminated data loss even at the highest frequency. This reduced both storage overhead and processing delays, ensuring high data integrity.
However, the extent of processing (using structures.py), along with how Sqlite3 search queries (using DBMakerL2.py) can be relatively slow, resulted in significant data loss. A backlog would form and eventually become too large for the computer to handle, resulting in the data loss. To fix this, I started collecting my data in batches, resulting in only one insert call being used as opposed to 1000 insert and searches to group data. This eliminated the data loss issue even at the highest possible 120Hz send rate, while still letting me process these custom structures of data.


## Phase 2: Model Selection and Training:

Relevant files: makeFeaturesFile.py, MLTime.py, FeaturePredictor.py

Once the data was collected, it was processed and prepared for model training. Given the limited amount of meaningful data (only around 150 laps of useful data), I engineered the dataset to extract lap-specific features for better prediction accuracy, I added a column to my dataframe that showed the time it took for the car to go from any one point on the lap, beck to the same point a lap later. By structuring the dataset to treat every frame as a lap in its own right, I expanded the dataset to hundreds of thousands of data points.

Several models were tested, including Decision Trees, XGBoost, and Neural Networks (using TensorFlow), before settling on RandomForestRegressor due to its resistance to overfitting with large datasets. Using feature importance from scikit-learn, I refined the model to improve performance further, choosing and removing features with more knowledge on their value.



## Phase 3: Race Strategy Generation


Relevant files: StrategyFinder.py

This phase focused on generating an optimal race strategy using a genetic algorithm. Nodes representing different race strategies were evaluated based on their performance, with the best nodes being replicated to refine the strategy further. This process was repeated across generations until an optimal pit strategy was identified.

Key challenges included designing the decision-making model for pit stops and tire selection. A softplus activation function was used to increase the likelihood of pitting as tire wear increased. Multiprocessing was employed to speed up the simulation of multiple generations, reducing computation time.



## Future of the Project
I plan to enhance my model by collecting more data, expanding its applicability to tracks beyond the Austrian F1 circuit, and improving its performance in varied weather conditions. Additionally, I aim to incorporate factors such as yellow and red flags into the model, allowing for more dynamic and realistic race strategies.

Looking ahead, I also intend to develop an AI chatbot that will serve as a virtual race engineer. This chatbot would provide real-time status updates, tweak strategies during the race, and offer personalized feedback to further improve the race experience.
