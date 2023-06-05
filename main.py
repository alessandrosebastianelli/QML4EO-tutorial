# Suppressing warning
import warnings
warnings.filterwarnings('ignore')

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    fxn()
    
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)

import numpy as np
np.seterr(all="ignore")

from utils.DatasetHandler import DatasetHandler
from qc.QiskitCircuit import QiskitCircuit
from models.HybridNet import HybridNet
from models.PyTorchModel import PyTorchModel

# Configuration file, please read it carefully
from config import *

import torch.optim as optim
import torch.nn as nn
import torch
import os


################################ Initialize Dataset Handler ################################
################################ and print classes          ################################
print('Loading Dataset')
dh = DatasetHandler(DATASET_ROOT)

classes = []
for i, c in enumerate(dh.classes):
    cl = c.split(os.path.sep)[-1]
    classes.append(cl)
classes.sort()
print('[*] Classes: {}'.format(classes))

################################ Load image paths and labels ################################
imgs, labels = dh.load_paths_labels(DATASET_ROOT, classes=classes)
print('[*] Size: {}'.format(len(imgs)))

################################# Training-Validation Split #################################
tra_imgs, tra_lbls, val_imgs, val_lbls = dh.train_validation_split(imgs, labels, SPLIT_FACTOR)
print('[*] Training Size:   {}'.format(len(tra_imgs)))
print('[*] Validation Size: {}'.format(len(val_imgs)))

###################################### Initialized QCNN #####################################
print('Initialize Quantum Hybrid Neural Network')
circuit = QiskitCircuit()
network = HybridNet()

optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)
criterion = nn.CrossEntropyLoss()

print('Printing Quantum Circuit')
print(circuit.circuit.draw(output='text', scale=1/NUM_LAYERS))

print('Printing Quantum Circuit Parameters')
print('[*] Number of Qubits:   {}'.format(NUM_QUBITS))
print('[*] Number of R Layers: {}'.format(NUM_LAYERS))
print('[*] Number of Outputs:  {}'.format(NUM_QC_OUTPUTS))
print('[*] Number of Shots:    {}'.format(NUM_SHOTS))

# This class wrap a PyTorch model. It is only needed to mask basic function, like model training.
model = PyTorchModel(network, criterion, optimizer)


###################################### Train QCNN #####################################
tra_set = [tra_imgs, tra_lbls]
val_set = [val_imgs, val_lbls]
model.fit(EPOCHS, tra_set, val_set, classes, batch_size=BATCH_SIZE, es=None, tra_size = None, val_size = None)