import numpy as np
import os

#----------------------- Quantum Circuit Settings -----------------------
NUM_QUBITS      = 4
NUM_SHOTS       = 1 # for timing reasons is set to 1, but in IRL you want this value to be higher https://quantumcomputing.stackexchange.com/questions/9823/what-is-meant-with-shot-in-quantum-computation
NUM_LAYERS      = 2
SHIFT           = np.pi/4

#----------------------- Dataset Settings -----------------------
DATASET_ROOT    = 'EuroSAT_2'
SPLIT_FACTOR    = 0.2

CLASS_DICT      = {
    "AnnualCrop":           0,
    "Forest":               1,
    "HerbaceousVegetation": 2,
    "Highway":              3,
    "Industrial":           4,
    "Pasture":              5,
    "PermanentCrop":        6,
    "Residential":          7,
    "River":                8,
    "SeaLake":              9
}


#----------------------- Training Settings -----------------------
TRAINING        = True
LOAD_CHECKPOINT = False
EPOCHS          = 20
LEARNING_RATE   = 0.002
MOMENTUM        = 0.5
BATCH_SIZE      = 1
CLASSES         = len(CLASS_DICT)

