						3D-Printer Experiment

preprocessing.py -- Does data preparation. Peak detection and data segmentation to feed into the classification algorithm. (Requires: preprocess(filename) -> truth values and split times

train.py -- Does training, imports preprocessing package saves the ML model specified by the user

reconstruction.py -- Does the reconstruction, imports the model from classification and raw data files from csv and wav and outputs the model of the printed structure
