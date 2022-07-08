"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

This is where you actually train models. See below for docs:

Generate a synthetic graph, with holes in upper left and lower right regions, + paths over the graph:
    python3 synthetic_data_gen.py
    -Edit main of synthetic_data_gen.py to change graph size / # of paths

Train a SCoNe model on a dataset:
    python3 trajectory_experiments.py [args]

    -Command to run standard training / experiment with defaults:
        python3 trajectory_experiments.py -data_folder_suffix suffix_here

    -The default hyperparameters should work pretty well on the default graph size. You'll probably have to play with
        them for other graphs, though.




Arguments + default values for trajectory_experiments.py:
   'model': 'scone'; which model to use, of ('scone', 'ebli', or 'bunch')
        -'scone': ours
        -'ebli':  https://arxiv.org/pdf/2010.03633.pdf
        -'bunch': https://arxiv.org/pdf/2012.06010.pdf
   'epochs': 1000; # of training epochs
   'learning_rate': 0.001; starting learning rate
   'weight_decay': 0.00005; ridge regularization constant
   'batch_size': 100; # of samples per batch (randomly selected)
   'reverse': 0;  if 1, also compute accuracy over the test set, but reversed (Reverse experiment)
   'data_folder_suffix': 'schaub2'; set suffix of folder to import data from (trajectory_data_Nhop_suffix)
   'regional': 0; if 1, trains a model over upper graph region and tests over lower region (Transfer experiment)

   'hidden_layers': 3_16_3_16_3_16 (corresponds to [(3, 16), (3, 16), (3, 16)]; each tuple is a layer (# of shift matrices, # of units in layer) )
        -'scone' and 'ebli' require 3_#_3_#_ ...; 'bunch' requires 7_#_7_#_ ...
   'describe': 1; describes the dataset being used
   'load_data': 1; if 0, generate new data; if 1, load data from folder set in data_folder_suffix
   'load_model': 0; if 0, train a new model, if 1, load model from file model_name.npy. Must set hidden_layers regardless of choice
   'markov': 0; include tests using a 2nd-order Markov model
   'model_name': 'model'; name of model to use when load_model = 1

   'flip_edges': 0; if 1, flips orientation of a random subset of edges. with tanh activation, should perform equally

   'multi_graph': '': if not '', also tests on paths over the graph with the folder suffix set here
   'holes': 1; if generation new data, sets whether the graph should have holes

More examples:
    python3 trajectory_experiments.py -model_name tanh -reverse 1 -epochs 1100 -load_model 1 -multi_graph no_holes
        -loads model tanh.npy from models folder, tests it over reversed test set, and also tests over another graph saved in trajectory_data_Nhop_no_holes
    python3 trajectory_experiments.py load_data 0 -holes 0 -model_name tanh_no_holes -hidden_layers [(3, 32), (3,16)] -data_folder_suffix no_holes2
        -generates a new graph with no holes; saves dataset to trajectory_data_Nhop_no_holes2;
            trains a new model with 2 layers (32 and 16 channels, respectively) for 1100 epochs, and saves its weights to tanh_no_holes.npy
    python3 trajectory_experiments.py -load_data 0 -holes 1 -data_folder_suffix holes
        -make a dataset with holes, save with folder suffix holes (just stop the program once training starts if you just want to make a new dataset)
    python3 trajectory_experiments.py load_data 0 -holes 0 -data_folder_suffix no_holes -model_name tanh_no_holes -multi_graph holes
        -create a dataset using folder suffix no_holes, train a model over it using default settings, and test it over the graph with data folder suffix holes
"""
import os, sys
import numpy as onp
from numpy import linalg as la
import jax.numpy as np
from jax.scipy.special import logsumexp




try:
    from trajectory_analysis.bunch_model_matrices import compute_shift_matrices
    from trajectory_analysis.synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from trajectory_analysis.scone_trajectory_model import Scone_GCN
    from trajectory_analysis.markov_model import Markov_Model
except Exception:
    from bunch_model_matrices import compute_shift_matrices
    from synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from scone_trajectory_model import Scone_GCN
    from markov_model import Markov_Model


def hyperparams():
    """
    Parse hyperparameters from command line

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'model': 'scnn2',
                   'epochs': 20,
                   'learning_rate': 0.001,
                   'weight_decay': 0.00005,
                   'batch_size': 100,
                   'hidden_layers': [(3, 16), (3, 16), (3, 16)], # where 3 indicates the 1 identity, 1 lower and 1 upper shift # for ebli, replace 3 by 4, for bunch, replace 3 by 7, for scnn, no need to replace
                   'k1_scnn': 2,
                   'k2_scnn': 2,
                   'describe': 1,
                   'reverse': 1,
                   'load_data': 1,
                   'load_model': 0,
                   'markov': 0,
                   'model_name': 'model',
                   'regional': 0,
                   'flip_edges': 0,
                   'data_folder_suffix': 'working',
                   'multi_graph': '',
                   'holes': 1}

    for i in range(len(args) - 1):
        if args[i][0] == '-':
            if args[i][1:] == 'hidden_layers':
                nums = list(map(int, args[i + 1].split("_")))

                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]
            elif args[i][1:] in ['model_name', 'data_folder_suffix', 'multi_graph', 'model']:
                hyperparams[args[i][1:]] = str(args[i+1])
            else:
                hyperparams[args[i][1:]] = float(args[i+1])


    return hyperparams

HYPERPARAMS = hyperparams()

### Model definition ###

# Activation functions
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.where(x >= 0, x, 0.01 * x)

# SCoNe function
def scone_func(weights, S_lower, S_upper, Bcond_func, last_node, flow):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / 3
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    print(S_lower.size)
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 3] \
                  + S_lower @ cur_out @ weights[i*3 + 1] \
                  + S_upper @ cur_out @ weights[i*3 + 2]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    #print(logits)
    return logits - logsumexp(logits) # log of the softmax function 

# SCNN with order K
# def scnn_func(weights, *shifts, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
#     """
#     Forward pass of the SCoNe model with variable number of layers
#     """
#     n_layers = (len(weights) - 1) / (1+k1+k2)
#     print('#weights:',len(weights),'#layers:',n_layers)
#     assert n_layers % 1 == 0, 'wrong number of weights'
#     cur_out = flow
#     for i in range(int(n_layers)):
#         cur_out = cur_out @ weights[i*(1+k1+k2)] 
#         cur_out.sum(shifts[k] @ cur_out @ weights[i*(1+k1+k2) + k] for k in len(shifts)) 

#         cur_out = tanh(cur_out)

#     logits = Bcond_func(last_node) @ cur_out @ weights[-1]
#     return logits - logsumexp(logits)

def scnn_func_2(weights, S_lower, S2_lower, S_upper, S2_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S_upper @ cur_out @ weights[i*n_k + 3] \
                  + S2_upper @ cur_out @ weights[i*n_k +4]    

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_3(weights, S_lower, S2_lower, S3_lower, S_upper, S2_upper, S3_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=3
    n_k = 1+k1+k2
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S_upper @ cur_out @ weights[i*n_k + 4] \
                  + S2_upper @ cur_out @ weights[i*n_k +5] \
                  + S3_upper @ cur_out @ weights[i*n_k + 6]       

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_4(weights, S_lower, S2_lower, S3_lower, S4_lower, S_upper, S2_upper, S3_upper, S4_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=4
    n_k = 1+k1+k2
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S4_upper @ cur_out @ weights[i*n_k + 4] \
                  + S_upper @ cur_out @ weights[i*n_k + 5] \
                  + S2_upper @ cur_out @ weights[i*n_k + 6] \
                  + S3_upper @ cur_out @ weights[i*n_k + 7] \
                  + S4_upper @ cur_out @ weights[i*n_k + 8]         

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

# Ebli function
def ebli_func(weights, S, S2, S3, Bcond_func, last_node, flow):
    """
    Forward pass of the Ebli model with variable number of layers
    note that here 
    S_lower = L1
    S_upper = L1^2
    """
    n_layers = (len(weights) - 1) / 4
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 4] \
                  + S @ cur_out @ weights[i*4 + 1] \
                  + S2 @ cur_out @ weights[i*4 + 2] \
                  + S3 @ cur_out @ weights[i*4 + 3]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

# Bunch function
def bunch_func(weights, S_00, S_10, S_01, S_11, S_21, S_12, S_22, nbrhoods, last_node, flow):
    """
    Forward pass of the Bunch model with variable number of layers
    """
    n_layers = (len(weights)) / 7
    print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = S_00 @ cur_out[0] @ weights[i * 7] \
                   + S_10 @ cur_out[1] @ weights[i * 7 + 1]

        next_out[1] = S_01 @ cur_out[0] @ weights[i * 7 + 2] \
                   + S_11 @ cur_out[1] @ weights[i * 7 + 3] \
                   + S_21 @ cur_out[2] @ weights[i * 7 + 4]

        next_out[2] = S_12 @ cur_out[1] @ weights[i * 7 + 5] \
                   + S_22 @ cur_out[2] @ weights[i * 7 + 6]

        cur_out = [relu(c) for c in next_out]

    nodes_out = cur_out[0] # use the last layer output on the node level as the final output 
    # values at nbrs of last node
    logits = nodes_out[nbrhoods[last_node]]
    return logits - logsumexp(logits)


def data_setup(hops=(1,), load=True, folder_suffix='schaub'):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
        at once
    """

    inputs_all, y_all, target_nodes_all = [], [], []

    if HYPERPARAMS['flip_edges']:
        # Flip orientation of a random subset of edges
        onp.random.seed(1)
        _, _, _, _, _, G_undir, _, _ = load_dataset('trajectory_data_1hop_' + folder_suffix)
        flips = onp.random.choice([1, -1], size=len(G_undir.edges), replace=True, p=[0.8, 0.2])
        F = np.diag(flips)


    if not load:
        # Generate new data
        generate_dataset(400, 1000, folder=folder_suffix, holes=HYPERPARAMS['holes'])
        raise Exception('Data generation done')


    for h in hops:
        # Load data
        folder = 'trajectory_data_' + str(h) + 'hop_' + folder_suffix
        X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)
        B1, B2 = B_matrices
        target_nodes_all.append(target_nodes)


        inputs_all.append([None, onp.array(last_nodes), X])
        y_all.append(y)

        # Define shifts
        L1_lower = B1.T @ B1
        L1_upper = B2 @ B2.T
        if HYPERPARAMS['flip_edges']:
            L1_lower = F @ L1_lower @ F
            L1_upper = F @ L1_upper @ F


        if HYPERPARAMS['model'] == 'scone':
            shifts = [L1_lower, L1_upper]
            # shifts = [L1_lower, L1_lower]
        
        elif HYPERPARAMS['model'] == 'scnn2':
            # shifts = [la.matrix_power(L1_lower,i+1) for i in range(HYPERPARAMS['k1_scnn'])]
            # shifts.append(la.matrix_power(L1_upper,i+1) for i in range(HYPERPARAMS['k2_scnn']))
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper, L1_upper@L1_upper]
            #L1 = L1_lower + L1_upper
            #shifts = [L1, L1 @ L1, L1@L1@L1,L1@L1@L1@L1] # test order 4 ebli _func 
            
        elif HYPERPARAMS['model'] == 'scnn3':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower,L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]    

        elif HYPERPARAMS['model'] == 'scnn4':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper] 
            
        elif HYPERPARAMS['model'] == 'ebli':
            L1 = L1_lower + L1_upper
            shifts = [L1, L1 @ L1, L1@L1@L1] # L1, L1^2

        elif HYPERPARAMS['model'] == 'bunch':
            # S_00, S_01, S_01, S_11, S_21, S_12, S_22
            shifts = compute_shift_matrices(B1, B2)

        else:
            raise Exception('invalid model type')

    # Build E_lookup for multi-hop training
    e = onp.nonzero(B1.T)[1]
    edges = onp.array_split(e, len(e)/2)
    E, E_lookup = [], {}
    for i, e in enumerate(edges):
        E.append(tuple(e))
        E_lookup[tuple(e)] = i

    # set up neighborhood data
    last_nodes = inputs_all[0][1]

    max_degree = max(G_undir.degree, key=lambda x: x[1])[1]
    nbrhoods_dict = {node: onp.array(list(map(int, G_undir[node]))) for node in
                     map(int, sorted(G_undir.nodes))}
    n_nbrs = onp.array([len(nbrhoods_dict[n]) for n in last_nodes])

    # Bconds function
    nbrhoods = np.array([list(sorted(G_undir[n])) + [-1] * (max_degree - len(G_undir[n])) for n in range(max(G_undir.nodes) + 1)])
    nbrhoods = nbrhoods

    # load prefixes if they exist
    try:
        prefixes = list(np.load('trajectory_data_1hop_' + folder_suffix + '/prefixes.npy', allow_pickle=True))
    except:
        prefixes = [flow_to_path(inputs_all[0][-1][i], E, last_nodes[i]) for i in range(len(last_nodes))]

    B1_jax = np.append(B1, np.zeros((1, B1.shape[1])), axis=0)

    if HYPERPARAMS['flip_edges']:
        B1_jax = B1_jax @ F
        for i in range(len(inputs_all)):
            print(inputs_all[i][-1].shape)
            n_flows, n_edges = inputs_all[i][-1].shape[:2]
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges)) @ F
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges, 1))

    def Bconds_func(n):
        """
        Returns rows of B1 corresponding to neighbors of node n
        """
        Nv = nbrhoods[n]
        return B1_jax[Nv]

    for i in range(len(inputs_all)):
        if HYPERPARAMS['model'] != 'bunch':
            inputs_all[i][0] = Bconds_func
        else:
            inputs_all[i][0] = nbrhoods
    
    return inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes

##
def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """

    # load dataset
    inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes = data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])

    (inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all
    #print(len(inputs_1hop), len(y_1hop))
    last_nodes = inputs_1hop[1]

    in_axes = tuple(([None] * len(shifts)) + [None, None, 0, 0])

    # Train Markov model
    if HYPERPARAMS['markov'] == 1:
        order = 1
        markov = Markov_Model(order)
        paths = onp.array([prefix + [target1, target2] for prefix, target1, target2 in zip(prefixes, target_nodes_all[0], target_nodes_all[1])], dtype='object')

        paths_train = paths[train_mask == 1]
        prefixes_train, target_nodes_1hop_train, target_nodes_2hop_train = onp.array(prefixes)[train_mask == 1], target_nodes_all[0][train_mask == 1], target_nodes_all[1][train_mask == 1]
        prefixes_test, target_nodes_1hop_test, target_nodes_2hop_test = onp.array(prefixes, dtype='object')[test_mask == 1], target_nodes_all[0][test_mask == 1], target_nodes_all[1][test_mask == 1]

        # forward paths
        markov.train(G_undir, paths_train)
        print("train accs")
        print(markov.test(prefixes_train, target_nodes_1hop_train, 1))
        print(markov.test(prefixes_train, target_nodes_2hop_train, 2))
        print(markov.test_2_target(prefixes_train, target_nodes_1hop_train))
        print("test accs")
        print(markov.test(prefixes_test, target_nodes_1hop_test, 1))
        print(markov.test(prefixes_test, target_nodes_2hop_test, 2))
        print(markov.test_2_target(prefixes_test, target_nodes_1hop_test))



        # reversed test paths
        rev_paths = [path[::-1] for path in paths]
        rev_prefixes = onp.array([p[:-2] for p in rev_paths], dtype='object')
        rev_prefixes_test = rev_prefixes[test_mask == 1]
        rev_target_nodes_1hop, rev_target_nodes_2hop = onp.array([p[-2] for p in rev_paths], dtype='object'), onp.array([p[-1] for p in rev_paths], dtype='object')
        rev_target_nodes_1hop_test = rev_target_nodes_1hop[test_mask == 1]
        rev_target_nodes_2hop_test = rev_target_nodes_2hop[test_mask == 1]
        print("Reversed test accs")
        print(markov.test(rev_prefixes_test, rev_target_nodes_1hop_test, 1))
        print(markov.test(rev_prefixes_test, rev_target_nodes_2hop_test, 2))

        # half forward, half backward
        fwd_mask = onp.array([True] * int(len(paths) / 2) + [False] * int(len(paths) / 2))
        onp.random.shuffle(fwd_mask)
        bkwd_mask = ~fwd_mask

        # mixed dataset
        mixed_paths = onp.concatenate((onp.array(paths)[fwd_mask == 1], onp.array(rev_paths)[bkwd_mask == 1]))
        mixed_prefixes = onp.concatenate((onp.array(prefixes)[fwd_mask==1], rev_prefixes[bkwd_mask==1]))
        mixed_target_nodes_1hop = onp.concatenate((target_nodes_all[0][fwd_mask == 1], rev_target_nodes_1hop[bkwd_mask == 1]))
        mixed_target_nodes_2hop = onp.concatenate((target_nodes_all[1][fwd_mask == 1], rev_target_nodes_2hop[bkwd_mask == 1]))

        # train / test splits
        mixed_paths_train = mixed_paths[train_mask == 1]
        mixed_prefixes_train, mixed_prefixes_test = mixed_prefixes[train_mask == 1], mixed_prefixes[test_mask == 1]
        mixed_target_nodes_1hop_train, mixed_target_nodes_1hop_test = mixed_target_nodes_1hop[train_mask == 1], mixed_target_nodes_1hop[test_mask == 1]
        mixed_target_nodes_2hop_train, mixed_target_nodes_2hop_test = mixed_target_nodes_2hop[train_mask == 1], mixed_target_nodes_2hop[test_mask == 1]

        markov.train(G_undir, mixed_paths_train)

        print("Mixed train accs")
        print(markov.test(mixed_prefixes_train, mixed_target_nodes_1hop_train, 1))
        print(markov.test(mixed_prefixes_train, mixed_target_nodes_2hop_train, 2))
        print("Mixed test accs")
        print(markov.test(mixed_prefixes_test, mixed_target_nodes_1hop_test, 1))
        print(markov.test(mixed_prefixes_test, mixed_target_nodes_2hop_test, 2))

        # train on middle, test on middle
        mid_train_mask = [i % 3 == 0 and train_mask[i] == 1 for i in range(len(train_mask))]
        mid_test_mask = [i % 3 == 0 and test_mask[i] == 1 for i in range(len(test_mask))]

        mid_paths_train, mid_paths_test = paths[mid_train_mask], paths[mid_test_mask]
        mid_prefixes_train, mid_prefixes_test = [p[:-2] for p in mid_paths_train], [p[:-2] for p in mid_paths_test]

        mid_targets_1hop_train, mid_targets_1hop_test = target_nodes_all[0][mid_train_mask], target_nodes_all[0][mid_test_mask]
        mid_targets_2hop_train, mid_targets_2hop_test = target_nodes_all[1][mid_train_mask], target_nodes_all[1][mid_test_mask]

        markov.train(G_undir, mid_paths_train)
        print("Middle region train accs")
        print(markov.test(mid_prefixes_train, mid_targets_1hop_train, 1))
        print(markov.test(mid_prefixes_train, mid_targets_2hop_train, 2))
        print("Middle region test accs")
        print(markov.test(mid_prefixes_test, mid_targets_1hop_test, 1))
        print(markov.test(mid_prefixes_test, mid_targets_2hop_test, 2))


        # train on upper, test on lower
        paths_upper = [paths[i] for i in range(len(paths)) if i % 3 == 1]
        prefixes_upper = [p[:-2] for p in paths_upper]
        targets_1hop_upper = [target_nodes_all[0][i] for i in range(len(paths)) if i % 3 == 1]
        targets_2hop_upper = [target_nodes_all[1][i] for i in range(len(paths)) if i % 3 == 1]

        paths_lower = [paths[i] for i in range(len(paths)) if i % 3 == 2]
        prefixes_lower = [p[:-2] for p in paths_lower]
        targets_1hop_lower = [target_nodes_all[0][i] for i in range(len(paths)) if i % 3 == 2]
        targets_2hop_lower = [target_nodes_all[1][i] for i in range(len(paths)) if i % 3 == 2]

        markov.train(G_undir, paths_upper)
        print("Upper region train accs")
        print(markov.test(prefixes_upper, targets_1hop_upper, 1))
        print(markov.test(prefixes_upper, targets_2hop_upper, 2))
        print("Lower region accs")
        print(markov.test(prefixes_lower, targets_1hop_lower, 1))
        print(markov.test(prefixes_lower, targets_2hop_lower, 2))
        raise Exception

    # Initialize model
    scone = Scone_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'], HYPERPARAMS['weight_decay'])

    if HYPERPARAMS['model'] == 'scone':
        model_func = scone_func
    elif HYPERPARAMS['model'] == 'ebli':
        model_func = ebli_func
    elif HYPERPARAMS['model'] == 'bunch':
        model_func = bunch_func
    elif HYPERPARAMS['model'] == 'scnn2':
        model_func = scnn_func_2
    elif HYPERPARAMS['model'] == 'scnn3':
        model_func = scnn_func_3
    elif HYPERPARAMS['model'] == 'scnn4':
        model_func = scnn_func_4
    else:
        raise Exception('invalid model')


    if HYPERPARAMS['model'] == 'scnn2' or HYPERPARAMS['model'] == 'scnn3' or HYPERPARAMS['model'] == 'scnn4':
        scone.setup_scnn(model_func, HYPERPARAMS['hidden_layers'], HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn'], shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])
    else:
        scone.setup(model_func, HYPERPARAMS['hidden_layers'], shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])

    if HYPERPARAMS['regional']:
        # Train either on upper region only or all data (synthetic dataset)
        # 0: middle, 1: top, 2: bottom
        train_mask = np.array([1 if i % 3 == 1 else 0 for i in range(len(y_1hop))])
        test_mask = np.array([1 if i % 3 == 2 else 0 for i in range(len(y_1hop))])

    # describe dataset
    if HYPERPARAMS['describe'] == 1:
        print('Graph nodes: {}, edges: {}, avg degree: {}'.format(len(G_undir.nodes), len(G_undir.edges), np.average(np.array([G_undir.degree[node] for node in G_undir.nodes]))))
        print('Training paths: {}, Test paths: {}'.format(train_mask.sum(), test_mask.sum()))
        print('Model: {}'.format(HYPERPARAMS['model']))

    # load a model from file + train it more
    if HYPERPARAMS['load_model']:
        if HYPERPARAMS['regional']:
            scone.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(HYPERPARAMS['epochs']) + '_regional' + '.npy', allow_pickle=True)
            print('load successful')
        else:
            scone.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(HYPERPARAMS['epochs']) + '.npy', allow_pickle=True)
            print('load successful')
        # if HYPERPARAMS['epochs'] != 0:
        #     # train model for additional epochs
        #     scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)
        #     try:
        #         os.mkdir('models')
        #     except:
        #         pass
        #     onp.save('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(HYPERPARAMS['epochs']), scone.weights)

        (test_loss, test_acc) = scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)
        print('test successful')
    else:

        train_loss, train_acc, test_loss, test_acc = scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)

        try:
            os.mkdir('models')
        except:
            pass
        if HYPERPARAMS['regional']:
            onp.save('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(HYPERPARAMS['epochs']) + '_regional', scone.weights)
        else: 
            onp.save('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(HYPERPARAMS['epochs']), scone.weights)

    # standard experiment
    print('standard test set:')
    scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)
    
    train_2target, test_2target = scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, train_mask, n_nbrs), scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, test_mask, n_nbrs)

    print('2-target accs:', train_2target, test_2target)


    if HYPERPARAMS['reverse']:
        # reverse direction of test flows
        rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
            onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), \
            onp.load('trajectory_data_2hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
        rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
        print('Reverse experiment:')
        scone.test([inputs_1hop[0], rev_last_nodes, rev_flows_in], rev_targets_1hop, test_mask, rev_n_nbrs)



    # print('Multi hop accs:',
    #       scone.multi_hop_accuracy_dist(shifts, inputs_1hop, target_nodes_all[1], [train_mask, test_mask], nbrhoods,
    #                                     E_lookup, last_nodes, prefixes, 2))


if __name__ == '__main__':
    train_model()