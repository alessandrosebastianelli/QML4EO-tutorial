from utils.DatasetHandler import DatasetHandler

from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import print_CF
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.auto import tqdm
import torch

class PyTorchModel:
    '''
        Wrapper for Pytorch Model. It contains basic functions, for training
        evaluation and plotting.
    '''

    def __init__(self, model, criterion, optimizer):
        '''
            PyTorchModel constructor

            Parameters
            ----------
            - model: pytorch model
            - criterion: pytorch loss function
            - optimizer: pytorch optimizer

            Returs
            ------
            Nothig, it created a PyTorchModel object
        '''

        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer

        # Initialize training loss and validation loss to zero (needed for tqdm)
        self.tra_losses = [0] + []
        self.val_losses = [0] + []

        # Initialize DatasetHandler to none, we need only the generator functions
        self.dh = DatasetHandler(None)

    def summary(self, shape):
        '''
            Keras-like model.summary, tt prints the summary of a model

            Parameters
            ----------
            - shape: input shape (without batch)
            
            Returns
            -------
            Nothing, it prints the suummary of the model
        '''
        print(summary(self.model, shape))
    
    def curves(self):
        '''
            Plot the training-validation curves

            Parameters
            ----------
            Nothing, it uses the global variables self.tra_losses and self.val_losses

            Returns
            -------
            Nothinh, it plots the training-validation curves
        '''
        fig = plt.figure()
        plt.plot(self.tra_losses[1:], '-*', label = 'training')
        plt.plot(self.val_losses[1:], '-*', label = 'validation')
        plt.legend()
        plt.show()
        
    def fit(self, epochs, tra_set, val_set, classes, batch_size=1, es=None, tra_size = None, val_size = None):
        '''
            Keras-like model.fit (or .fit_generator), it trains and validates a pytorch model

            Parameters
            ----------
            - epochs: number of training epochs
            - tra_set: tuple [img_paths, img_labels] containing training image paths and training image labels
            - val_set: tuple [img_paths, img_labels] containing validation image paths and validation image labels
            - classes: list of string containing class names
            - batch size [default 1]: size of batch for training. N.B. for hybrid quantum neural network must stay to 1
            - es: Early Stoppin, must be one of the two defined in utils/EarlyStopping.py
            - tra_size [default None]: keras-like steps_per_epoch, if none it is the size of the training dataset
            - val_size [default None]: keras-like validation_steps, if none it is the size of the validation dataset

            Returns
            -------
            Nothing, it trains the model
        '''

        ## Unpack training set
        tra_imgs = tra_set[0]
        tra_lbls = tra_set[1]

        ## Upack validation set
        val_imgs = val_set[0]
        val_lbls = val_set[1]

        ## Calulate the training and validation steps
        if tra_size == None: tra_size = len(tra_imgs)
        if val_size == None: val_size = len(val_imgs)


        pb = tqdm(range(epochs), dynamic_ncols=True) # Initialize training progress bar
        esct = 0 # Early stopping counter, please refers to utils/EarlyStopping.py

        for epoch in pb:
            ## Initialize training and validation data loader
            tra_dl  = iter(self.dh.data_loader(tra_imgs, tra_lbls, batch_size=batch_size, img_shape=(64,64,3)))
            val_dl  = iter(self.dh.data_loader(val_imgs, val_lbls, batch_size=batch_size, img_shape=(64,64,3)))
            
            ## ------------------------------------- Training ------------------------------------- 
            tra_loss = 0.0
            val_loss = 0.0

            for i in range(tra_size//batch_size):
                x, y    = next(tra_dl)
                outputs = self.model.forward(x)
                print(outputs)
                self.optimizer.zero_grad()
                loss    = self.criterion(outputs, y) 
                loss.backward() 
                self.optimizer.step() 

                l = loss.item()
                tra_loss += l
                pb.set_description("Train - [E %d/%d s %d] [B %d/%d - B Loss %1.4f] - T Loss %1.4f - V Loss %1.4f" % (epoch+1, epochs, esct+1, i+1, tra_size//batch_size, l, self.tra_losses[-1], self.val_losses[-1]))
            
            self.tra_losses.append(tra_loss/i)
            
            ## ------------------------------------- Validation ------------------------------------- 
            pb2 = tqdm(range(val_size//batch_size), leave=False, dynamic_ncols=True)
            with torch.no_grad():
                targets = []
                predictions = []

                for i in pb2:
                    x, y    = next(val_dl)
                    outputs = self.model.forward(x)
                    loss    = self.criterion(outputs, y)
                    for i in range(len(y)):
                        targets.append(y[i])
                        predictions.append(torch.argmax(outputs[i,...]))
                    l = loss.item()
                    val_loss += l

                    pb2.set_description("Valid - [E %d/%d s %d] [B %d/%d - B Loss %1.4f] - T Loss %1.4f - V Loss %1.4f" % (epoch+1, epochs, esct+1, i+1, val_size//batch_size, l, self.tra_losses[-1], self.val_losses[-1]))
                
                self.val_losses.append(val_loss/i)
            
            ## Early Stopping
            if es!=None:
                es(self.tra_losses[-1], self.val_losses[-1])
                if es.early_stop:
                    break
                esct = es.counter

            ## Every 3 epochs print the classification report
            if epoch == 0:
                cf = confusion_matrix(targets, predictions, normalize='true')
                cr = classification_report(targets, predictions, target_names=classes, digits=4)
                print('\t\t [*] Confusion Matrix:')
                print_CF(cf, classes)
                print('\t\t [*] Classification Report:')
                print(cr)
            ## Every 3 epochs print the classification report
            if epoch % 3 == 1:
                cf = confusion_matrix(targets, predictions, normalize='true')
                cr = classification_report(targets, predictions, target_names=classes, digits=4)
                print('\t\t [*] Confusion Matrix:')
                print_CF(cf, classes)
                print('\t\t [*] Classification Report:')
                print(cr)