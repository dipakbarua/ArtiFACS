#!/usr/bin/python
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.stats import norm
import plot_accuracy
import plot_marker_wavelength_distributions 
import plot_fingerprint 
import plot_cell_marker_expression
import plot_scatter_protein_expression 
import plot_detector_signal_distributions  
from math import * 

def phi(x, m, s): 
   
    # 'Calculates cdf' 
 
    return (1.0 + erf((x-m)/(s*sqrt(2.0))))/2.0   


def pr(m, s, lx, ux): 
    # 'probability calculated between lx and ux'

    return (phi(ux, m, s) - phi(lx, m, s))

def plot_histogram(mean, std):
    data = np.random.normal(mean, std, 10000000)
    plt.hist(data,1000, alpha = 0.5, density=True)
        
 
class Parameters:
    def __init__(self, n_cell):
        self.n_channels = []
        self.n_cell = n_cell


class Channel:
    def __init__(self, mu, bandpass):
        self.mu = mu
        self.bandpass = bandpass
        self.lower = mu - bandpass/2.0
        self.upper = mu + bandpass/2.0
          
class Marker: 
    def __init__(self, name, mu, sigma, rel_intensity):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.rel_intensity = rel_intensity
        self.influence_factors = []
    
    def compute_influence_factors(self, channels): 
        for i in range(len(channels)): 
            chl = channels[i].lower
            chu = channels[i].upper
            prob = pr(self.mu, self.sigma, chl, chu)
            self.influence_factors.append(self.rel_intensity*prob)  
     

class Cell:
    def __init__(self, cell_type, ID): 
        self.cell_type = cell_type
        self.ID = ID
        self.intensity = [] # one intensity value for each marker 
        self.fingerprint = {} 

    def generate_fingerprint(self, markers):
        n_ch = len(markers[0].influence_factors) # Get the number of channels 

        V = [] 

        for i in range(n_ch):
            sum_ = 0 
            for j in range(len(markers)):
                inten = self.intensity[j] 
                influ = markers[j].influence_factors[i] 
                sum_ = sum_ + inten*influ 
            
            V.append(sum_)

        self.fingerprint[0] = V 


class Cell_type:


    def __init__(self, type_name, n_cell, markers):
        self.type_name = type_name
        self.n_cell = n_cell
        self.markers = markers
        self.logM = np.random.uniform(0.0, 2.3026,len(markers)) 
        self.sig = np.random.uniform(0.01, 1.0, len(markers))   
        self.cells = [] 
               
    def create_cells(self):
        for i in range(self.n_cell):
            c = Cell(self.type_name, i)
            for j in range(len(self.logM)):
                mu_ = self.logM[j] 
                std_ = self.sig[j]  
                logValue_ = np.random.normal(mu_,std_)
                value_ = np.exp(logValue_)   
                c.intensity.append(value_) 

            self.cells.append(c) 

 
class Sample:
    def __init__(self):
        self.n_cell_type = 0
        self.n_cell = 0 
        self.cell_types = []
         

def define_markers(marker_info): 
    markers = [] 
    
    base = marker_info[0] 

    diff = marker_info[1] #50 
    
    stdv = marker_info[2]    # 30 

    n_marker = marker_info[3] 
   
    rel_intensity = 1 

    for i in range(n_marker):
        mean = base + (i+1)*diff 
        name = "marker" + str(i+1)
        markers.append(Marker(name, mean, stdv, rel_intensity))

    return markers

   
     

def define_real_markers(): 
    markers = [] 

    stdv = 30  
    

    mean = 584
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 0.3 
    markers.append(Marker('mEos2', mean, stdv, rel_intensity)) 
    
    mean = 470
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 1.0 
    markers.append(Marker('hypothetical', mean, stdv, rel_intensity))

    
    mean = 519  
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 0.65 
    markers.append(Marker('Alexa Fluor 488', mean, stdv, rel_intensity))
 
    mean = 595   
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 0.1 
    markers.append(Marker('PA-mCherry1', mean, stdv, rel_intensity))
    

    mean = 670   
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 0.7 
    markers.append(Marker('Cy5', mean, stdv, rel_intensity))

    plt.show() 


    return markers

def define_autofluorescence_markers(): 
    markers = [] 

    mean = 340
    stdv = 30  
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 10 
    markers.append(Marker('tryptophan', mean, stdv, rel_intensity)) 
      
    
    mean = 390 
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 5 
    markers.append(Marker('pyridoxin', mean, stdv, rel_intensity))

    
    mean = 470  
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 3 
    markers.append(Marker('NADH', mean, stdv, rel_intensity))
 
    mean = 550   
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 2 
    markers.append(Marker('riboflavins', mean, stdv, rel_intensity))
    

    mean = 660   
    #stdv = np.random.uniform(0, 10.0) 
    rel_intensity = 1 
    markers.append(Marker('porphyrins', mean, stdv, rel_intensity))

    return markers


class FlowCyt:

    markers = [] 
 
    def __init__(self):
        self.channels = []
        self.sample = Sample()  

    def create_channels(self,l_lambda, u_lambda, n_channel, bandpass): 
        mean_detection_lambda = np.linspace(l_lambda, u_lambda, n_channel) 
   
        #print(mean_detection_lambda) 
 
        for i in range(n_channel):
            ch = Channel(mean_detection_lambda[i], bandpass)
            self.channels.append(ch) 
    
     
    def create_sample(self, n_cell_type, n_cell, markers):
        self.sample.n_cell_type = n_cell_type
        self.sample.n_cell = n_cell 

        
        for i in range(n_cell_type):
            ctype = Cell_type(i, n_cell, markers)
            ctype.create_cells()
            self.sample.cell_types.append(ctype)      

    def compute_influence_factors(self, markers, channels):
        for i in range(len(markers)): 
            markers[i].compute_influence_factors(channels)  
             

    def generate_fingerprints(self):
        n_cell_type = len(self.sample.cell_types)
        n_cell = len(self.sample.cell_types[0].cells)
        n_channel = len(self.channels) 

        X = np.zeros((n_cell_type, n_cell, n_channel))
 
        for i in range(n_cell_type):
            for j in range(n_cell):
                self.sample.cell_types[i].cells[j].generate_fingerprint(self.markers)
                for k in range(n_channel):
                    X[i,j,k] = self.sample.cell_types[i].cells[j].fingerprint[0][k] 

        return X

def get_training_and_test_sets(sample, n_training_instance, n_test_instance): 
    n_cell_types = len(sample.cell_types) 
    
    train_sample = [None for i in range(n_cell_types)] 
    test_sample = [None for i in range(n_cell_types)]

    for i in range(n_cell_types):
        train_sample[i] = sample.cell_types[i].cells[0:n_training_instance]   
        test_sample[i] = sample.cell_types[i].cells[n_training_instance:n_training_instance + n_test_instance] 

    train_set = [train_sample[i] for i in range(n_cell_types)]   
    test_set = [test_sample[i] for i in range(n_cell_types)]  

    train_set = np.concatenate(train_set) 
    test_set = np.concatenate(test_set)
    
    train_cells = shuffle(train_set, random_state=0) 
    test_cells = shuffle(test_set, random_state=0)


    fingerprint_size = len(train_set[0].fingerprint[0])

    X_train = np.zeros((len(train_cells),1,fingerprint_size))
    y_train = np.zeros((len(train_cells),1))
    X_test = np.zeros((len(test_cells),1,fingerprint_size))
    y_test = np.zeros((len(test_cells),1))

    for i in range(len(train_cells)):
        for j in range(1): 
            for k in range(fingerprint_size):
                X_train[i,j,k] = train_cells[i].fingerprint[j][k] 
                
    for i in range(len(train_cells)):
        y_train[i] = train_cells[i].cell_type  
    

    for i in range(len(test_cells)):
        for j in range(1): 
            for k in range(fingerprint_size):
                X_test[i,j,k] = test_cells[i].fingerprint[j][k]  
    
    for i in range(len(test_cells)):
        y_test[i] = test_cells[i].cell_type



    return X_train, y_train, X_test, y_test, train_cells, test_cells  


      

def create_dnn(input_shape_X, input_shape_Y, n_hidden_node, n_output_node):
    dnn = keras.Sequential()
    dnn.add(keras.layers.Flatten(input_shape=(input_shape_X, input_shape_Y)))
    dnn.add(keras.layers.Dense(n_hidden_node, activation='relu'))
    dnn.add(keras.layers.Dense(n_hidden_node, activation='relu'))
    dnn.add(keras.layers.Dense(n_hidden_node, activation='relu'))
    dnn.add(keras.layers.Dense(n_output_node, activation=tf.nn.softmax))
    return dnn


def compile_dnn(dnn): 

    dnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return dnn 



def fit_dnn(dnn, X_train, y_train, epochs): 
    dnn.fit(X_train, y_train, epochs=epochs)
    return dnn 



def test_accuracy(dnn, X_test, y_test): 

    test_loss, test_acc = dnn.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)


def make_predictions(dnn, X_test, y_test, test_sample, type_of_interest=None, cutoff=None): 
    predictions = dnn.predict(X_test)

    m, n, k = X_test.shape
 
    X_ = np.empty((1,1,k)) 

    TP = 0 
    TN = 0
    FP = 0
    FN = 0 

    correct = 0
    incorrect = 0

    n_cell_cutoff = 0 
    correct_cutoff = 0
    incorrect_cutoff = 0
    dumped = 0 

    #predictions_1 = dnn.predict(X_test[0]) 
    print("prediction : ", predictions[1])

    n_test_cell = len(test_sample) 


    cells_correct_above_cutoff = []
    cells_incorrect_above_cutoff = []
    cells_correct_below_cutoff = []  
    cells_below_cutoff = []  

    for i in range(n_test_cell):
        X_[0] = X_test[i]  
        pred = dnn.predict(X_)   
        maxWeight = np.amax(pred)


        if (type_of_interest):
            type_of_interest = type_of_interest
        else:
            type_of_interest = 0  # Default value for cell type of interest 


        threshold = pred[0][type_of_interest]

        #print(threshold, '   ', pred[0]) 

        
        if (cutoff):
            cutoff = cutoff 
        else:
            cutoff = 0 # default value 


        if (threshold > cutoff):
            if ((y_test[i] == type_of_interest)):
                TP = TP + 1 
            
            if ((y_test[i] != type_of_interest)):
                FP = FP + 1
            
        else:
            if ((y_test[i] == type_of_interest)):
                FN = FN + 1
            
            if ((y_test[i] != type_of_interest)):
                TN = TN + 1



        actual = y_test[i] 
        prediction = np.argmax(pred[0])


        
        if (maxWeight > cutoff): 
            print(i, "    ", 'actual: ', actual, '      ', 'prediction: ', prediction, '       ', 'Accuracy: ', maxWeight)
            n_cell_cutoff = n_cell_cutoff + 1 
            if (actual == prediction):
                cells_correct_above_cutoff.append(test_sample[i]) 
                correct_cutoff = correct_cutoff + 1
            else:
                cells_incorrect_above_cutoff.append(test_sample[i])  
                incorrect_cutoff = incorrect_cutoff + 1 
        else:
            if (actual == prediction):
                cells_correct_below_cutoff.append(test_sample[i])  
            print(i, "    ", 'actual: ', actual, 'Dumped', 'prediction: ', prediction, '       ', 'Accuracy: ', maxWeight)
            dumped = dumped + 1 
       
        if (actual == prediction):   
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    plot_scatter_protein_expression.plot_(cells_correct_above_cutoff, cutoff)

    TPR = TP/(TP + FN) 

    FPR = FP/(FP + TN)

    accuracy = 1.0*correct/n_test_cell

    
    purity = len(cells_correct_above_cutoff)/(len(cells_incorrect_above_cutoff) + len(cells_correct_above_cutoff)) 

    #purity = correct_cutoff/(correct_cutoff + incorrect_cutoff) 


    fraction_dumped = dumped/n_test_cell 

    return TPR, FPR, accuracy, purity, fraction_dumped, cells_correct_above_cutoff, cells_incorrect_above_cutoff   

def evaluate_roc(dnn, X_test, y_test, test_sample, type_of_interest, file_name): 
    f = open(file_name, "a")
    
    type_of_interest = 0

    cutoff = 0.40

    for i in range(300): 
        #cutoff = cutoff/3.0  
        cutoff = cutoff - 0.001   
        TPR, FPR, accuracy, purity, fraction_dumped, cells_correct_above_cutoff, cells_incorrect_above_cutoff \
            = make_predictions(dnn, X_test, y_test, test_sample, type_of_interest, cutoff)
        f.write("%f,%f\n"% (FPR, TPR)) 

    f.close()


def evaluate_accuracy(dnn, X_test, y_test, test_sample):
    
    #f = open(filename, "a")
    
    TPR, FPR, accuracy, purity, fraction_dumped, cells_correct_above_threshold, cells_incorrect_above_threshold \
            = make_predictions(dnn, X_test, y_test, test_sample)
    #f.write("%f\n"% (accuracy)) 

    #f.close()

    return accuracy

def print_accuracy(data, n_channel, marker_mean, marker_stdv, filename, write_header):
    
    f = open(filename, "a")
   
    if (write_header == 1):
        f.write("%s,%s,%s,%s,%s\n"% ("n_channel,","Delta m", "s", "accuracy", "std"))   
 
    f.write("%f,%f,%f,%f,%f\n"% (n_channel, marker_mean, marker_stdv, np.mean(data.accuracy), np.std(data.accuracy)))  

    f.close()

def evaluate_purity(dnn, X_test, y_test, test_sample, file_name, cutoffs):
    
    f = open(file_name, "a")
  
    #cutoffs = [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8] 

    cutoffs = cutoffs

 
    for i in range(len(cutoffs)):
        cutoff = cutoffs[i]  
        TPR, FPR, accuracy, purity, fraction_dumped, cells_correct_above_threshold, cells_incorrect_above_threshold \
             = make_predictions(dnn, X_test, y_test, test_sample, cutoff=cutoff)

        
        f.write("%f,%f,%f\n"% (cutoff,purity, fraction_dumped)) 

    f.close()


def get_cell_and_flowcytometer_data(n_channel, n_cell_type, n_cell, marker_info): 
    
    bandpass = 30 

    fw = FlowCyt() 

    fw.create_channels(400, 900, n_channel, bandpass)

    fw.markers = define_markers(marker_info)
    
    #plot_marker_wavelength_distributions.plot_(fw.markers) 

    fw.compute_influence_factors(fw.markers, fw.channels)  
    
    fw.create_sample(n_cell_type, n_cell, fw.markers)

    X = fw.generate_fingerprints()
    
    return X, fw  


class Data:
    def __init__(self,n_iter):
        self.accuracy = np.zeros(n_iter)


