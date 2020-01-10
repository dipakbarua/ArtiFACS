#!/usr/bin/python
import model


def main():

    #################### BEGIN PARAMETER BLOCK ###############################
   
   
    #********** Manuscript parameter: N **************************
    n_channels = ([16])   # Manuscript variable: N 
    #n_channels = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20])
    
    
    #********** Manuscript parameter: C **************************
    n_cell_type = 4       # Manuscript variable: M
    
    
    #********** Manuscript parameter: M **************************
    n_marker = 5 

    #********** Manuscript parameter: Delta m **************************
    marker_mean_diff = ([50]) # Manuscript variable Delta M 
    #marker_mean_diff = ([0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    
    #********** Manuscript parameter: s **************************
    marker_stdv = ([30])  # Manuscript variable s 
    #marker_stdv = ([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
   
    #********** Manuscript parameter: H **************************
    cutoffs = [0] # Manuscript variable H  
    #cutoffs = [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8] 
    
    #################### END PARAMETER BLOCK ###############################
    
    
    n_cell = 22000
    marker_base = 400
    m_training_instance = 20000
    m_test_instance = n_cell - m_training_instance 
    

    #marker_mean_diff = np.random.uniform(1, 50, 1000)
    #marker_stdv = (np.random.uniform(0.1, 3,1000))**3 

    n_iter = 1    # Set this value to take the average accuracy calculations from multiple runs  

    write_header = 1 

    file_accuracy = "accuracy.csv"

    for i in range(len(n_channels)): # loop over the channels  
        
        n_ch = n_channels[i] 
    
        for mm in range(len(marker_mean_diff)):   # loop over the markers 
            for k in range(len(marker_stdv)):

                marker_info = []
                marker_info.append(marker_base)
                marker_info.append(marker_mean_diff[mm])
                marker_info.append(marker_stdv[k])
                marker_info.append(n_marker) 

                data = model.Data(n_iter)  

                for j in range(n_iter):     # iterations for the same set of channels and markers (however creates a new set of cells everytime)  
          
                    X, fw = model.get_cell_and_flowcytometer_data(n_ch, n_cell_type, n_cell, marker_info)
             
                   
                    
                    n_cell, n_channel_used = X[0].shape #  
                   
                    #model.plot_cell_marker_expression.plot_(fw.sample, fw.markers) # Uncomment if you would like to see he marker expression plot 
                    #model.plot_fingerprint.plot_(X)                                # Uncomment if you would like to see the fingerprints  
                    #model.plot_detector_signal_distributions.plot_(X)              # Uncomment if you would like to see signal distribution in every detector  

                 
                    X_train, y_train, X_test, y_test, train_sample, test_sample = \
                            model.get_training_and_test_sets(fw.sample, m_training_instance, m_test_instance) 
    
  
                    n_hidden_layer_node = 100
                    n_output_layer_node = n_cell_type

                    input_shape_X = 1  # 1D image (only 1 pixel in the vertical direction) 
                    input_shape_Y = n_channel_used # Number of pixes in the horizontal direction 
                    
                    dnn = model.create_dnn(input_shape_X, input_shape_Y, \
                            n_hidden_layer_node, n_output_layer_node) 
                    
                    dnn = model.compile_dnn(dnn) 
                   
                    epochs = 10 

                    dnn = model.fit_dnn(dnn, X_train, y_train, epochs) 

                    
                    type_of_interest = 0 

                    #file_name = "roc_0_" + str(n_ch) + ".csv" 
                    
                    #model.evaluate_roc(dnn, X_test, y_test, test_sample, type_of_interest, file_name) 
                    
                    file_name = "purity_0_" + str(n_ch) + ".csv" 

                    #model.evaluate_purity(dnn, X_test, y_test, test_sample, file_name, cutoffs)
                    
                    data.accuracy[j] = model.evaluate_accuracy(dnn, X_test, y_test, test_sample) 

                model.print_accuracy(data, n_ch, marker_mean_diff[mm], marker_stdv[k], file_accuracy, write_header)

                write_header = 0 
 
    #model.plot_accuracy.plot_()



if __name__ == "__main__":
    main() 
