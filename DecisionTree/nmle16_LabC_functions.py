import numpy as np
from collections import deque
import copy

def import_training_set(data_filename):
    """Import the set of examples from text file
    Assumption about text file:
    Each line: attributes then classification all split by space --> tuples

    Input: data file name
    Output:
    1. Tuples of attributes 
    2. List of example tuples
    """
    with open(data_filename,'r') as data_file:
        all_lines = data_file.read().split('\n')
    attributes = tuple(all_lines[0].split('\t'))
    examples_list = []
    for i in all_lines[1:]:
        if len(i)>0:
            examples_list.append(tuple(i.split('\t')))
    return attributes, examples_list     

def single_set_entropy(example_set):
    """Calculate the entropy of an example set
    Assumption about the example_set:
    A list of example tuples, with the last element of a tuples being classification
    Input: example set
    Return: entropy of that set
    """
    no_examples = len(example_set)
    positive = 0
    for i in example_set:
        if i[-1] == 'yes':
            positive +=1
    if positive == no_examples or positive == 0:
        return 0
    positive_probability = positive/no_examples
    entropy = -(positive_probability*np.log2(positive_probability)+
                (1- positive_probability)*np.log2(1-positive_probability))
    return entropy

def gain_one_attribute(example_set, attribute_index):
    """Calculate information gain from splitting on an attribute
    Input: example set, tuple index of the attribute to be splitted on
    Output: A list of
    1. information gain 
    2. a list of distinct values in the example set for that attribute
    3. a list of list of examples corresponds to the distinct values
    4. the index of the attribute
    """
    # calculate total entropy
    total_entropy = single_set_entropy(example_set)
    # split examples into subset based on distinct values of the attribute
    distinct_values = [] #contains a list of distinct values for this attribute
    distinct_splitting = []  #contains the splitting lists based on distinct values
    for example in example_set:
        if example[attribute_index] not in distinct_values:
            distinct_values.append(example[attribute_index])
            distinct_splitting.append([])
        splitting_index = distinct_values.index(example[attribute_index])
        distinct_splitting[splitting_index].append(example)
    # calculate the remaining entropy
    remainder = 0 
    for subset in distinct_splitting:
        subset_entropy = single_set_entropy(subset)
        weight = len(subset)/len(example_set)
        remainder += weight*subset_entropy
    # calculate information gain
    information_gain = total_entropy - remainder
    return [information_gain, distinct_values, distinct_splitting, attribute_index]        

# functions to handle the output of gain_one_attribute
def get_info_gain(gain_split):
    return gain_split[0]
def get_distinct_values(gain_split):
    return gain_split[1]
def get_splitting(gain_split):
    return gain_split[2]
def get_split_variable(gain_split):
    return gain_split[3]

def plurality(example_set):
    """Calculate the plurality of an example set
    Input: example set
    Output: "yes" or "no" which ever appears more times in the classification of the examples"""
    positive = 0
    for i in example_set:
        if i[-1] == 'yes':
            positive += 1
    if positive/len(example_set) > 0.5:
        return 'yes'
    #also return no if tie
    else:
        return 'no'

class Node():
    def __init__(self,example_set, attributes = None, split_variable = None, distinct_value = None, decision = None,classification = None):
        self.example_set = example_set 
        self.attributes = attributes    # list of available attribute indices to split
        self.split_variable = None      # split variable for the example set at that node
        self.distinct_value = distinct_value    # distinct value at that node corresponding to split_variable of parent
        self.decision = decision        # indicator of a leaf
        self.classification = classification    # classification of an object if the node is a leaf
        self.children = []
        self.parent = None
        self.depth = 0 
    def set_split_variable(self, split_variable):
        self.split_variable = split_variable

def split_node(node,frontier):
    """Expand a node/do splitting for maximum information gain"""
    gains = []
    info_gains =[]
    for i in node.attributes:
        single_gain_split = gain_one_attribute(node.example_set,i)
        gains.append(single_gain_split)
        info_gains.append(get_info_gain(single_gain_split))
    # get the attribute splitting with maximum information gain
    max_index = info_gains.index(max(info_gains))
    max_gain_split = gains[max_index] #list of [gain, distinct_values, splitting, split_attribute_index]
    
    # handle case: splitting gives no gain in information
    if get_info_gain(max_gain_split) <= 0:
        node.decision = True 
        node.classification = plurality(node.example_set) 
    
    #set split variable for parent node
    node.set_split_variable(get_split_variable(max_gain_split))
    
    #create children nodes
    splitting = get_splitting(max_gain_split)               # list of subsets by distinct_values
    distinct_values = get_distinct_values(max_gain_split)   # list of distinct_values
    for i in range(len(distinct_values)):
        remaining_attributes = copy.deepcopy(node.attributes)
        remaining_attributes.remove(node.split_variable)
        new_node = Node(splitting[i],remaining_attributes,None,distinct_values[i],None, None)
        
        # handle case: run out of attributes to split on
        if len(remaining_attributes) == 0:
            new_node.decision = True
            new_node.classification = plurality(new_node.example_set)
        # handle case: new node has only positive or only negative examples
        elif single_set_entropy(new_node.example_set) == 0:
            new_node.decision = True
            if new_node.example_set[0][-1] == 'yes':
                new_node.classification = 'yes'
            else:
                new_node.classification = 'no'
        # append the new node to the frontier otherwise
        else:
            frontier.append(new_node)
        # link parent and children
        new_node.parent = node
        new_node.depth = new_node.parent.depth + 1
        node.children.append(new_node)
        
def display_tree(root_node, attributes):
    """Take in root of a tree and list of attributes to display that tree
    using depth first search traversal"""
    traverse_frontier = []
    traverse_frontier.append(root_node)
    num_nodes = 0
    while len(traverse_frontier) != 0:
        current_node = traverse_frontier.pop()
        num_nodes +=1 
        print_string = '\t'*current_node.depth
        if current_node.distinct_value != None:
            print_string = print_string+ '='+str(current_node.distinct_value)+':'
        if current_node.split_variable != None:
            print_string = print_string +'-->' +str(attributes[current_node.split_variable])+':'
        if current_node.decision == True:
            print_string = print_string +'-->' + current_node.classification
        print(print_string)
        for i in current_node.children:
            traverse_frontier.append(i)    
    print('Total number of nodes: '+ str(num_nodes))

def predict(root_node, example):
    current_node = root_node
    while current_node.decision == None:
        list_distinct_values = []
        for child_node in current_node.children:
            list_distinct_values.append(child_node.distinct_value)
        # handle case: example has a new value to the tree 
        if example[current_node.split_variable] not in list_distinct_values:
            return plurality(current_node.example_set)
        # move to children with matching values recursively
        i = 0 
        while i < len(list_distinct_values):
            if list_distinct_values[i] == example[current_node.split_variable]:
                current_node = current_node.children[i]
                break
            i +=1
    return current_node.classification

def build_tree(attributes, example_list):
    """Build a tree from a list of attributes and a set of examples
    Return: the root node of the tree"""
    # initialize root node and frontier
    root_node = Node(example_list, list(range(len(attributes)-1)), None, None,None, None)
    frontier = deque()
    frontier.append(root_node)
    num_expanded_node = 0

    # build the tree
    while len(frontier) != 0:
        node_to_be_expanded = frontier.popleft()
        split_node(node_to_be_expanded, frontier)
        num_expanded_node +=1
    
    return root_node

