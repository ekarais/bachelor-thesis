#!/usr/bin/env python
# coding: utf-8

'''
Comments:
1. The panel frames are darker, almost black in PGM.
2. Nearby colors are indistinguishable to the human eye. Perhaps go coarser, 5 instead of 10?

Roadmap:
1. [DONE] Implement a single triple
2. [DONE] Implement all triples
3. [DONE]Implement all combinations of triples
'''

import sys, getopt, os
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import random
import string
#np.random.seed(42)
#plt.rcParams["figure.figsize"] = (1,1)

def get_random_triples(num=None, seed=None):
    # set seed for debugging
    if seed is not None:
        np.random.seed(seed)
    
    # set the number of triples: uniformly sampled from {1, 2, 3, 4} as in the paper. Note that we could have up to 6 triples in a PGM.
    if num is None:
        num = 1 + np.random.choice(4)
    
    # flags will constrain the set of all triple combinations to the set of valid combinations. flags[i] == True iff a triple of the type map[i]
    # has not been selected yet.
    shape_position_X = False
    flags = [True, True, True, True, True, True]

    # mapping
    map = [['shape', 'size'],
        ['shape', 'color'],
        ['shape', 'position/number'],
        ['shape', 'type'],
        ['line', 'color'],
        ['line', 'type']]
        

    # to be returned
    triples = []

    # other initializations
    b_probs = [0.5, 0.5]

    for i in range(num):
        # select the "type" of the new triple uniformly at random from all the remaining types left.
        probs = np.multiply(flags, np.ones(6)/np.sum(flags))
        idx = np.random.choice(6, 1, p=probs)[0].astype(int)
        flags[idx] = False
        
        # handle the mutual exclusiveness of shape-position and shape-number.
        if map[idx][1] == 'position/number':
            
            if np.random.choice(2, p=b_probs):
                map[idx][1] = 'position'
                shape_position_X = True
            else:
                map[idx][1] = 'number'

        triple = [map[idx][0], map[idx][1]]

        # get the set of possible relations we can choose from to append to the chosen triple type
        relations = triples_[map[idx][0]][map[idx][1]]
        
        # determine the conditional distribution to sample from the set of relations, depending on
        # the type of the triple and the triples that we have already sampled.

        ## If a triple with type = line-position is already present, we only allow line-color-[progression|union]
        if triple == ['line', 'color'] and (not flags[5]):
            r_probs = [0.5, 0, 0, 0, 0.5]

        ## If a triple with type = shape-position is already present, we disallow triples of the form shape-[size|color|type]-[xor|or|and]
        elif triple[0] == 'shape' and triple[1] in ['size', 'color', 'type'] and shape_position_X:
            r_probs = [0.5, 0, 0, 0, 0.5]
        
        else:
            r_probs = np.ones(len(relations)) / len(relations) #uniform
        
        # sample a relation according to the computed conditional distribution and append the complete
        # triple to the list of sampled triples.
        r_idx = np.random.choice(len(relations), 1, p=r_probs)[0].astype(int) #?
        relation = relations[r_idx]
        triple.append(relation)

        # Set flags and distributions to help determine the next triple.

        ## If a triple = line-color-[xor|or|and] is sampled, we disallow any triple with type line-position.
        if triple[:2] == ['line', 'color'] and triple[2] in ['XOR', 'OR', 'AND']:
            flags[5] = False
        
        ## If a triple with form shape-[size|color|type]-[xor|or|and] is sampled, we disallow triples with type = shape-position . 
        if (triple[0] == 'shape') and (triple[1] in ['size', 'color', 'type']) and (triple[2] in ['XOR', 'OR', 'AND']):
            b_probs = [1,0]

        
        triples.append(triple)

    return triples

def process_triples(triples):
    '''
    Organize the list of triples for downstream processing.
    '''
    shape_triples = []
    line_triples = []

    #separating shape generation from line generation
    for triple in triples:
        if triple[0] == 'shape':

            #putting number/position relation to the end
            if triple[1] in ['number', 'position']:
                shape_triples.append(triple)
            else:
                shape_triples.insert(0, triple)
        else:

            #putting type relation to the end
            if triple[1] == 'type':
                line_triples.append(triple)
            else:
                line_triples.insert(0, triple)

    return shape_triples, line_triples

def get_non_identical_sets(n, upper_bound=3):
    ''' 
    Returns non-identical sets where set elements are positive natural numbers (>0)
    '''
    sets = []
    for i in range(2):
        card = 2 + np.random.choice(upper_bound) #one element will be discarded later
        sets.append(np.random.choice(n, card, replace=False))
    
    #if sets are identical, swap last uncompared element in for the 1st element
    if set(sets[0][:sets[0].size-1]) == set(sets[1][:sets[1].size-1]):
        sets[1][0] = sets[1][sets[1].size-1]
    
    #discard the last elements
    sets[0] = sets[0][:sets[0].size-1]
    sets[1] = sets[1][:sets[1].size-1]

    return 1 + sets[0], 1 + sets[1]

def get_value_sets(relation, n, lb=1):
    '''
    Returns a list containing 9 numpy arrays.
    '''

    value_sets = [[], [], [], [], [], [], [], [], []]

    # u_values, permutations and three_unique_permutations are only used if relation = union.
    u_values = 1 + (lb-1) + np.random.choice(n-(lb-1), 3, replace=False) 
    permutations = np.array([[0, 1, 2],
                             [0, 2, 1],
                             [1, 0, 2],
                             [1, 2, 0],
                             [2, 0, 1],
                             [2, 1, 0]])
    three_unique_permutations = permutations[np.random.choice(6,3, replace=False)]

    values = [[], [], []]

    for i in range(3):

        if relation == 'progression':
            value_list = 1 + (lb-1) + np.sort(np.random.choice(n-(lb-1), 3, replace=False))
            for k in range(3):
                values[k] = np.array([value_list[k]])

        if relation == 'XOR':
            values[0], values[1] = get_non_identical_sets(n)
            #if the sets are disjunct, make them non-disjunct
            if len(set(values[0]).intersection(set(values[1]))) == 0:
            
                #if the sets are disjunct singletons, the modification in the else clause would make 'em identical
                if values[0].size == values[1].size and values[0].size == 1:
                    values[1] = np.concatenate((values[0], values[1]))
                else:
                    values[1][0] = values[0][0]

            values[2] = np.array(list(set(values[0]).symmetric_difference(set(values[1]))))
            
        if relation == 'OR':
            values[0], values[1] = get_non_identical_sets(n)
            #if the sets are disjunct, make them non-disjunct
            if len(set(values[0]).intersection(set(values[1]))) == 0:

                #if the sets are disjunct singletons, the modification in the else clause would make 'em identical
                if values[0].size == values[1].size and values[0].size == 1:
                    values[1] = np.concatenate((values[0], values[1]))
                else:
                    values[1][0] = values[0][0]
            values[2] = np.array(list(set(values[0]).union(set(values[1]))))

        if relation == 'AND':
            values[0], values[1] = get_non_identical_sets(n)
            #if the sets are disjunct, make them non-disjunct
            if len(set(values[0]).intersection(set(values[1]))) == 0:

                #if the sets are disjunct singletons, the modification in the else clause would make 'em identical
                if values[0].size == values[1].size and values[0].size == 1:
                    values[1] = np.concatenate((values[0], values[1]))
                else:
                    values[1][0] = values[0][0]
                
            values[2] = np.array(list(set(values[0]).intersection(set(values[1]))))

        if relation == 'union':
            value_list = u_values[three_unique_permutations[i]]
            for k in range(3):
                values[k] = np.array([value_list[k]])

        #randomizing the list order
        values[2] = values[2][np.random.choice(values[2].size, values[2].size, replace=False)]
        
        for j in range(3):
            value_sets[3*i+j] = values[j]

    return value_sets

def generate_RPM_(triples=None):
    '''
    Equivalent to generate_RPM(). The code is easier to understand, but has more lines.
    '''
    if triples is None:
        triples = get_random_triples()

    print(triples)
    shape_triples, line_triples = process_triples(triples)

    shape_values = [[],[],[]]

    for triple in shape_triples:
        if triple[1] in ['size', 'color', 'type']:
            n = attr_domains_[triple[1]]['shape'].size
            value_sets = get_value_sets(triple[2], n)

            if triple[1] == 'size':
                shape_values[0] = value_sets
            if triple[1] == 'color':
                shape_values[1] = value_sets
            if triple[1] == 'type':
                shape_values[2] = value_sets

    
    #pre-sampling constant values for size/color/type to be used if these attributes aren't present in the triples
    attribute_constants = [1+np.random.choice(attr_domains_['size']['shape'].size),
        1+np.random.choice(attr_domains_['color']['shape'].size),
        1+np.random.choice(attr_domains_['type']['shape'].size)]

    if shape_triples[-1][1] == 'number':
        num_elements = 0
        for value_sets in shape_values:
            for set in value_sets:
                if set.size > num_elements:
                    num_elements = set.size
        print("Lower bound = {}".format(num_elements))
        nums_per_panel = get_value_sets(shape_triples[-1][2], 9, lb=num_elements)
        positions = np.random.choice(9, 9, replace=False)
        shape_matrices = []
        for i in range(9):
            M = np.zeros((9,3))
            for a in range(3):
                if len(shape_values[a]) > 0:
                    values = shape_values[a][i]
                    print(values)
                    print("Num. elements = {}".format(nums_per_panel[i][0]))
                    #Assign each value to one object
                    M[positions[:values.size],a] = values
                    #For every remaining object, assign one value from the value set at random.
                    M[positions[values.size:nums_per_panel[i][0]],a] = values[np.random.choice(values.size, nums_per_panel[i][0] - values.size)]
                
                else:
                    M[positions[:nums_per_panel[i][0]],a] = attribute_constants[a]
                    
            print(M)
            shape_matrices.append(M.astype(int))


    elif shape_triples[-1][1] == 'position':
        
        positions_per_panel = get_value_sets(shape_triples[-1][2], 9)
        #Post-processing as position indices start from 0
        for i in range(9):
            positions_per_panel[i] = positions_per_panel[i] - 1

        #positions = np.random.choice(9, 9, replace=False)
        shape_matrices = []
        for i in range(9):
            M = np.zeros((9,3))
            for a in range(3):
                if len(shape_values[a]) > 0:
                    values = shape_values[a][i]
                    print(values)
                    print("Positions = {}".format(positions_per_panel[i]))
                    #Assign each value to one object
                    #M[positions[:values.size],a] = values
                    M[positions_per_panel[i][:values.size],a] = values
                    #For every remaining object, assign one value from the value set at random.
                    #M[positions[values.size:nums_per_panel[i][0]],a] = values[np.random.choice(values.size, nums_per_panel[i][0] - values.size)]
                    M[positions_per_panel[i][values.size:positions_per_panel[i].size],a] \
                     = values[np.random.choice(values.size, positions_per_panel[i].size - values.size)]
                
                else:
                    M[positions_per_panel[i],a] = attribute_constants[a]
                    
            print(M)
            shape_matrices.append(M.astype(int))
    
    else:
        num_elements = 0
        for value_sets in shape_values:
            for set in value_sets:
                if set.size > num_elements:
                    num_elements = set.size
        print(num_elements)
        positions = np.random.choice(9, num_elements, replace=False)
        shape_matrices = []
        for i in range(9):
            M = np.zeros((9,3))
            for a in range(3):
                if len(shape_values[a]) > 0:
                    values = shape_values[a][i]
                    print(values)
                    #Assign each value to one object
                    M[positions[:values.size],a] = values
                    #For every remaining object, assign one value from the value set at random.
                    M[positions[values.size:num_elements],a] = values[np.random.choice(values.size,num_elements - values.size)]
                
                else:
                    M[positions,a] = attribute_constants[a]
                    
            print(M)
            shape_matrices.append(M.astype(int))

    
    
    
    RPM = empty_RPM()
    for i in range(9):
        RPM[i]['shapes'] = shape_matrices[i]
        #RPM[i]['lines'] = line_vectors[i]
    return RPM

def generate_RPM(triples=None):
    
    # If no triples are passed to the function, generate your own.
    if triples is None:
        triples = get_random_triples()

    # Separate shape/line triples, insert special value if no shape/line triple exists.
    shape_triples, line_triples = process_triples(triples)
    shape_triples = [[-1, -1, -1]] if shape_triples == [] else shape_triples
    line_triples = [[-1, -1, -1]] if line_triples == [] else line_triples
    
    # shape_values[i][j][k] describes the value of attribute i for the k-th object in the j-th grid cell.
    shape_values = [[], [], []]

    # sampling values for triples of type (shape, [size|color|type], relation)
    for triple in shape_triples:
        if triple[1] in ['size', 'color', 'type']:
            # n is the cardinality of the set of values that size/color/type can take for shapes.
            n = attr_domains_[triple[1]]['shape'].size
            value_sets = get_value_sets(triple[2], n)

            if triple[1] == 'size':
                shape_values[0] = value_sets
            if triple[1] == 'color':
                shape_values[1] = value_sets
            if triple[1] == 'type':
                shape_values[2] = value_sets

    
    # pre-sampling constant values for size/color/type to be used if these attributes aren't present in the triples
    attribute_constants = [1+np.random.choice(attr_domains_['size']['shape'].size),
        1+np.random.choice(attr_domains_['color']['shape'].size),
        1+np.random.choice(attr_domains_['type']['shape'].size)]

    # sampling values for the triple of type (shape, [number|position], relation) if it exists
    ## num_element denotes the minimum number of objects that will be present in any grid cell. This number
    ## is lower-bounded by the cardinality of the largest value set.
    num_elements = 0
    for value_sets in shape_values:
        for set in value_sets:
            if set.size > num_elements:
                num_elements = set.size
    
    shape_matrices = []
    positions = np.random.choice(9, 9, replace=False)

    # Case 1: a triple (shape, number, relation) exists
    if shape_triples[-1][1] == 'number':
        values_per_panel = get_value_sets(shape_triples[-1][2], 9, lb=num_elements)
        positions_per_panel = [positions[:values_per_panel[i][0]] for i in range(9)]

    # Case 2: a triple (shape, position, relation) exists
    elif shape_triples[-1][1] == 'position':
        values_per_panel = get_value_sets(shape_triples[-1][2], 9)
        
        #Post-processing as position indices start from 0
        for i in range(9):
            values_per_panel[i] -= 1
        
        positions_per_panel = values_per_panel
    
    # Case 3: otherwise
    else:
        # making sure some shapes my appear even if there are no shape relations
        num_elements = np.random.choice(10) if num_elements == 0 else num_elements
        values_per_panel = [[num_elements] for i in range(9)]
        positions_per_panel = [positions[:values_per_panel[i][0]] for i in range(9)]

    for i in range(9):
        M = np.zeros((9, 3))
        for a in range(3):
            if len(shape_values[a]) > 0:
                values = shape_values[a][i]
                #Assign each value to one object
                M[positions_per_panel[i][:values.size], a] = values
                #For every remaining object, assign one value from the value set at random.
                M[positions_per_panel[i][values.size:positions_per_panel[i].size], a] \
                    = values[np.random.choice(values.size, positions_per_panel[i].size - values.size)]
                
            else:
                M[positions_per_panel[i], a] = attribute_constants[a]
                    
        
        shape_matrices.append(M.astype(int))

    #DEALING WITH THE LINE RELATIONS
    line_values = []
    if line_triples[0][1] == 'color':
        n = attr_domains_['color']['line'].size
        line_values = get_value_sets(line_triples[0][2], n)

    color_constant = 1 + np.random.choice(attr_domains_['color']['line'].size)

    if line_triples[-1][1] == 'type':
        types_per_panel = get_value_sets(line_triples[-1][2], 6)
        #Post-processing as position indices start from 0
        for i in range(9):
            types_per_panel[i] -= 1

        line_vectors = []
        for i in range(9):
            v = np.zeros(6)
            
            if len(line_values) > 0:
                values = line_values[i]
                #Assign each value to one object
                v[types_per_panel[i][:values.size]] = values
                #For every remaining object, assign one value from the value set at random.
                v[types_per_panel[i][values.size:types_per_panel[i].size]] \
                    = values[np.random.choice(values.size, types_per_panel[i].size - values.size)]
                
            else:
                v[types_per_panel[i]] = color_constant
                    
            
            line_vectors.append(v.astype(int))

    else:
        num_types_lb = 0
        for line_value in line_values:
            num_types_lb = line_value.size if line_value.size > num_types_lb else num_types_lb
        
        num_types = num_types_lb + np.random.choice(attr_domains_['type']['line'].size - num_types_lb + 1)
        types = np.random.choice(attr_domains_['type']['line'].size, num_types, replace=False)
        line_vectors = []
        for i in range(9):
            v = np.zeros(6)
            
            if len(line_values) > 0:
                values = line_values[i]
                #Assign each value to one object
                v[types[:values.size]] = values
                #For every remaining object, assign one value from the value set at random.
                v[types[values.size:num_types]] = values[np.random.choice(values.size, num_types - values.size)]
                
            else:
                v[types] = color_constant
                    
            
            line_vectors.append(v.astype(int))

    # concatenate shape and line instances in a single data structure.
    RPM = empty_RPM()
    for i in range(9):
        RPM[i]['shapes'] = shape_matrices[i]
        RPM[i]['lines'] = line_vectors[i]
    return RPM

def get_random_RPM(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    panel_list = []
    for i in range(9):
        line_vec = np.random.choice(2,6)
        for j in range(6):
            line_vec[j] = 1 + np.random.choice(10) if line_vec[j] == 1 else line_vec[j]
        shape_M = np.empty((9,3))
        exists = np.random.choice(2,9)
        shape_M[:,0] = ((np.ones(9) + np.random.choice(10,9)) * exists).astype(int)
        shape_M[:,1] = ((np.ones(9) + np.random.choice(10,9)) * exists).astype(int)
        shape_M[:,2] = ((np.ones(9) + np.random.choice(7,9)) * exists).astype(int)
        panel = {
            'shapes' : shape_M.astype(int),
            'lines'  : line_vec.astype(int)
        }
        panel_list.append(panel)
    return panel_list

def empty_RPM():
    panel_list = []
    for i in range(9):
        line_vec = np.zeros(6)
        shape_M = np.zeros((9,3)).astype(int)
        panel = {
            'shapes' : shape_M.astype(int),
            'lines'  : line_vec.astype(int)
        }
        panel_list.append(panel)
    return panel_list

def render(RPM, name=None, directory='./'):
    fig, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20,20))
    fig.set_dpi(40)
    c = 0
    area = 36*100
    for i in range(3):
        for j in range(3):

            ax[i,j].axis('scaled')
            ax[i,j].set_xlim([0,1])
            ax[i,j].set_ylim([0,1])
            ax[i,j].set_xticks([],[])
            ax[i,j].set_yticks([],[])

            for o in range(9):
                x, y = attr_domains['position'][o]
                exists = RPM[c]['shapes'][o,0]

                if(exists > 0):
                    size = attr_domains['size'][RPM[c]['shapes'][o,0] - 1]
                    color = str(attr_domains['color'][RPM[c]['shapes'][o,1] - 1])
                    marker = attr_domains['type']['shape'][RPM[c]['shapes'][o,2] - 1] 
                    ax[i,j].scatter(x, y, area*(0.5 + size), c=color, marker=marker, edgecolors='k')        

            if (RPM[c]['lines'][0] >= 1):
                idx = RPM[c]['lines'][0] - 1
                ax[i,j].plot([1, 0], [0, 1], c=str(attr_domains['color'][idx]), linewidth=5, zorder=0.99, transform=ax[i,j].transAxes)
            if (RPM[c]['lines'][1] >= 1):
                idx = RPM[c]['lines'][1] - 1
                ax[i,j].plot([0, 1], [0, 1], c=str(attr_domains['color'][idx]), linewidth=5, zorder=0.99, transform=ax[i,j].transAxes)
            if (RPM[c]['lines'][2] >= 1):
                idx = RPM[c]['lines'][2] - 1
                ax[i,j].axvline(0.5, 0, 1, c=str(attr_domains['color'][idx]), linewidth=5, zorder=0.99)
            if (RPM[c]['lines'][3] >= 1):
                idx = RPM[c]['lines'][3] - 1
                ax[i,j].axhline(0.5, 0, 1, c=str(attr_domains['color'][idx]), linewidth=5, zorder=0.99)
            if (RPM[c]['lines'][4] >= 1):
                idx = RPM[c]['lines'][4] - 1
                rec = mpatches.Rectangle((0.5, 0), np.sqrt(0.5), np.sqrt(0.5), angle=45, edgecolor=str(attr_domains['color'][idx]), linewidth=5, fill=False, zorder=0.9)
                ax[i,j].add_artist(rec)
            if (RPM[c]['lines'][5] >= 1):
                idx = RPM[c]['lines'][5] - 1
                circ = mpatches.Circle((0.5, 0.5), 0.25, edgecolor=str(attr_domains['color'][idx]), linewidth=5, fill=False, zorder=0.9)
                ax[i,j].add_artist(circ)

            c += 1
    if name is None:
        plt.savefig(directory + 'RPM.png')

    else:
        plt.savefig(directory + name + '.png')

def to_npvector(RPM, onehot=False):
    
    if onehot == False:
        M = np.empty((9,33))
        for i in range(9):
            s = RPM[i]['shapes'].flatten()
            l = RPM[i]['lines']
            M[i] = np.concatenate((s,l))
    
    if onehot == True:
        M = np.empty((9,336))
        for i in range(9):
            s1 = to_categorical(RPM[i]['shapes'][:,0], num_classes=11)
            s2 = to_categorical(RPM[i]['shapes'][:,1], num_classes=11)
            s3 = to_categorical(RPM[i]['shapes'][:,2], num_classes=8)
            l = to_categorical(RPM[i]['lines'], num_classes=11)
            M[i] = np.concatenate((s1.flatten(), s2.flatten(), s3.flatten(), l.flatten()))
    
    
    return M.flatten()

def logits_to_RPM(logits):
    RPM = empty_RPM()
    
    for i in range(9):
        panel = logits[i]
        sizes = panel[:99].reshape((9,11))
        colors = panel[99:198].reshape((9,11))
        types = panel[198:270].reshape((9,8))
        assert np.allclose(np.sum(sizes, axis=1), 1)
        assert np.allclose(np.sum(colors, axis=1), 1)
        assert np.allclose(np.sum(types, axis=1), 1)
        shape_M = np.empty((9,3))
        shape_M[:,0] = np.argmax(sizes, axis=1)
        shape_M[:,1] = np.argmax(colors, axis=1)
        shape_M[:,2] = np.argmax(types, axis=1)
        RPM[i]['shapes'] = shape_M.astype(int)

        lv = panel[-66:].reshape((6,11))
        assert np.allclose(np.sum(lv, axis=1), 1)
        line_vec = np.argmax(lv, axis=1)
        RPM[i]['lines'] = line_vec.astype(int)

    return RPM

def to_RPM(vec, onehot=False):

    if onehot == False:
        RPM = []
        for i in range(9):
            panel = vec[i*33:i*33+33]
            assert panel.size == 33
            shape_M = panel[:27].reshape((9,3))
            line_vec = panel[27:]
            RPM.append(dict([
                ('shapes', shape_M),
                ('lines', line_vec)
            ]))
    
    if onehot == True:
        RPM = []
        for i in range(9):
            panel = vec[i*336:i*336+336]
            assert panel.size == 336
            sizes = np.argmax(panel[:99].reshape((9,11)), axis=1)
            colors = np.argmax(panel[99:198].reshape((9,11)), axis=1)
            types = np.argmax(panel[198:270].reshape((9,8)), axis=1)
            line_vec = np.argmax(panel[270:336].reshape((6,11)), axis=1)
            shape_M = np.concatenate((sizes[np.newaxis,:], colors[np.newaxis,:], types[np.newaxis,:])).T
            RPM.append(dict([
                ('shapes', shape_M),
                ('lines', line_vec)
            ]))
    return RPM

def is_equal(RPM_1, RPM_2):
    equal = True
    
    for i in range(9):
        if not np.array_equal(RPM_1[i]['shapes'], RPM_2[i]['shapes']):
            equal = False
        if not np.array_equal(RPM_1[i]['lines'], RPM_2[i]['lines']):
            equal = False

    return equal

def assert_equal(RPM_1, RPM_2):
    
    for i in range(9):
        assert np.array_equal(RPM_1[i]['shapes'], RPM_2[i]['shapes'])
        assert np.array_equal(RPM_1[i]['lines'], RPM_2[i]['lines'])
    
def sort_triples(triples):
    ss = []
    sc = []
    st = []
    sn = []
    sp = []
    lc = []
    lt = []
    for triple in triples:
        if triple[0] == 'shape' and triple[1] == 'size':
            ss.append(triple)
        elif triple[0] == 'shape' and triple[1] == 'color':
            sc.append(triple)
        elif triple[0] == 'shape' and triple[1] == 'type':
            st.append(triple)
        elif triple[0] == 'shape' and triple[1] == 'number':
            sn.append(triple)
        elif triple[0] == 'shape' and triple[1] == 'position':
            sp.append(triple)
        elif triple[0] == 'line' and triple[1] == 'color':
            lc.append(triple)
        else:
            lt.append(triple)
    
    return ss + sc + st + sn + sp + lc + lt

relations = ['progression', 'XOR', 'OR', 'AND', 'union']
objects = ['shape', 'line']
attributes = ['size', 'color', 'number', 'position', 'type']

triples__ = [[objects[0], attributes[0], relations[0]], #shape, size, progression
            [objects[0], attributes[0], relations[1]],  #shape, size, XOR
            [objects[0], attributes[0], relations[2]],  #shape, size, OR
            [objects[0], attributes[0], relations[3]],  #shape, size, AND
            [objects[0], attributes[0], relations[4]],  #shape, size, union
            [objects[0], attributes[1], relations[0]],  #shape, color, progression
            [objects[0], attributes[1], relations[1]],  #shape, color, XOR
            [objects[0], attributes[1], relations[2]],  #shape, color, OR
            [objects[0], attributes[1], relations[3]],  #shape, color, AND
            [objects[0], attributes[1], relations[4]],  #shape, color, union
            [objects[0], attributes[2], relations[0]],  #shape, number, progression
            [objects[0], attributes[2], relations[4]],  #shape, number, union
            [objects[0], attributes[3], relations[1]],  #shape, position, XOR
            [objects[0], attributes[3], relations[2]],  #shape, position, OR
            [objects[0], attributes[3], relations[3]],  #shape, position, AND
            [objects[0], attributes[4], relations[0]],  #shape, type, progression
            [objects[0], attributes[4], relations[1]],  #shape, type, XOR
            [objects[0], attributes[4], relations[2]],  #shape, type, OR
            [objects[0], attributes[4], relations[3]],  #shape, type, AND
            [objects[0], attributes[4], relations[4]],  #shape, type, union
            [objects[1], attributes[1], relations[0]],  #line, color, progression
            [objects[1], attributes[1], relations[1]],  #line, color, XOR
            [objects[1], attributes[1], relations[2]],  #line, color, OR  
            [objects[1], attributes[1], relations[3]],  #line, color, AND
            [objects[1], attributes[1], relations[4]],  #line, color, union
            [objects[1], attributes[4], relations[1]],  #line, position, XOR
            [objects[1], attributes[4], relations[2]],  #line, position, OR
            [objects[1], attributes[4], relations[3]],  #line, position, AND
            [objects[1], attributes[4], relations[4]]]  #line, position, union

triples_ = dict([
    ('shape', dict([
        ('size', ['progression', 'XOR', 'OR', 'AND', 'union']),
        ('color', ['progression', 'XOR', 'OR', 'AND', 'union']),
        ('number', ['progression', 'union']),
        ('position', ['XOR', 'OR', 'AND']),
        ('type', ['progression', 'XOR', 'OR', 'AND', 'union'])
    ])),
    ('line', dict([
        ('color', ['progression', 'XOR', 'OR', 'AND', 'union']),
        ('type', ['XOR', 'OR', 'AND', 'union'])
    ]))
])

types = dict([
    ('shape', np.array(['o', '^', 's', 'p',
               'h', '8', '*'])),
    ('line', np.array(['diag down', 'diag up', 'vertical',
               'horizontal', 'diamond', 'circle']))
])

attr_domains = {
    attributes[0] : np.linspace(0, 1, num=10, endpoint=False),
    attributes[1] : np.linspace(0, 1, num=10, endpoint=False),
    attributes[2] : np.arange(10),
    attributes[3] : np.array([(0.25,0.75),(0.5,0.75),(0.75,0.75),
                             (0.25,0.5),(0.5,0.5),(0.75,0.5),
                             (0.25,0.25),(0.5,0.25),(0.75,0.25)]),
    attributes[4] : types
}

attr_domains_ = {
    attributes[0] : dict([
        ('shape', np.linspace(0, 1, num=10, endpoint=False))
    ]),
    attributes[1] : dict([
        ('shape', np.linspace(0, 1, num=10, endpoint=False)),
        ('line', np.linspace(0, 1, num=10, endpoint=False))
    ]),
    attributes[2] : dict([
        ('shape', np.arange(10))    
    ]),
    attributes[3] : dict([
        ('shape', np.array([(0.25,0.75),(0.5,0.75),(0.75,0.75),
                             (0.25,0.5),(0.5,0.5),(0.75,0.5),
                             (0.25,0.25),(0.5,0.25),(0.75,0.25)]))      
    ]),
    attributes[4] : types
}


def main(argv):
    
    # Set program variables onehot (flag that controls if the PGMs will be stored in one-hot format or in graded format)
    # and num_samples (controls the number of samples to be generated)
    onehot = False
    num_samples = 10
    opts, _ = getopt.getopt(argv, "on:", ["onehot", "num_samples="])
    for opt, arg in opts:
        if opt in ('-o', '--onehot'):
            onehot = True
        if opt in ('-h', '--num_samples'):
            num_samples = int(arg)

    # Create the directory where the generated dataset will be stored.
    data_directory_prefix = '/home/ege/Documents/bthesis/data/'

    if onehot:
        data_directory_suffix = 'onehot/neutral_' + str(num_samples)
    else:
        data_directory_suffix = 'graded/neutral_' + str(num_samples)

    data_directory = data_directory_prefix + data_directory_suffix + '/'
    
    
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    for _ in tqdm(range(num_samples)):
        triples = sort_triples(get_random_triples())
        ###if ['shape', 'type', 'progression'] in triples:
        ###    print('present!')
        # The name of a particular PGM has the following template:
        # The first 3*n_i characters encode the triples that are present in the PGM, where the triples are sorted as defined in 
        # sort_triples(), and n_i denotes the number of triples that are present in the i-th PGM. This is followed by an underscore
        # and a 20-digit random digit so that PGMs with the exact same set of rules do not overwrite each other. 
        RPM_name = ''
        for triple in triples:
            RPM_name += str(objects.index(triple[0]))
            RPM_name += str(attributes.index(triple[1]))
            RPM_name += str(relations.index(triple[2]))
        rand_suffix = string.ascii_lowercase
        RPM_name += '_' + ''.join(random.choice(rand_suffix) for j in range(20))
        
        # Generate a PGM instance which contains the triples sampled by get_random_triples().
        RPM = generate_RPM(triples)
        vec = to_npvector(RPM, onehot=onehot)
        
        # Encode the triples present in the PGM; these serve as the labels in our training pipeline.
        triples_oh = np.zeros(29)
        for triple in triples:
            triples_oh[triples__.index(triple)] = 1
        
        ###if triples_oh[15] == 1:
        ###    print('also present!')
        # Save the generated PGM and the triples it contains to the data directory.
        np.save(data_directory+RPM_name, np.concatenate((vec,triples_oh)))
    
    triples = [['shape', 'color', 'union'], ['line', 'type', 'XOR']]
    RPM = generate_RPM(triples)
    render(RPM)
    
if __name__ == "__main__":
    main(sys.argv[1:])





