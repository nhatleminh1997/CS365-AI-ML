from nmle16_LabC_functions import*
import numpy as np
from collections import deque
import copy 
import sys

if __name__ == '__main__':
    # import the data
    attributes, example_list = import_training_set(sys.argv[1]) 

    # build and display the tree with full data set
    root_node = build_tree(attributes,example_list)
    display_tree(root_node, attributes)
    print("Total number of example used to train: "+str(len(example_list)))
    
    # whole training set accuracy
    ts_correctness = []
    for i in range(len(example_list)):
        prediction = predict(root_node, example_list[i])
        if prediction == example_list[i][-1]:
            ts_correctness.append(1)
        else:
            ts_correctness.append(0)
    ts_accuracy = sum(ts_correctness)/len(ts_correctness)
    print("Training set accuracy: " + str(ts_accuracy*100)+'%')
    
    # Leave-one-out Cross validation
    loocv_correctness = []
    for i in range(len(example_list)):
        train_list = example_list[:i] + example_list[i+1:]
        tree_root = build_tree(attributes, train_list)
        prediction = predict(tree_root, example_list[i])
        if prediction == example_list[i][-1]:
            loocv_correctness.append(1)
        else: 
            loocv_correctness.append(0)
    loocv_accuracy = sum(loocv_correctness)/len(loocv_correctness)
    print ('LOOCV accuracy: '+str(loocv_accuracy*100)+'%')