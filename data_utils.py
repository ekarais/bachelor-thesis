import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import warnings

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import grid
from tqdm import tqdm

class RegeneratedPGM(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = os.listdir(self.data_path)
        self.length = len(self.data)
        print("Initialized dataset with ", self.length, " samples.")

    def __getitem__(self, index):
        item = np.load(self.data_path + self.data[index]).astype(int)
        return torch.from_numpy(item[:-29]), torch.from_numpy(item[-29:]) 

    def __len__(self):
        return self.length


class NoUnionPGM(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = os.listdir(self.data_path)
        self.length = len(self.data)
        print("Initialized dataset with ", self.length, " samples.")

    def __getitem__(self, index):
        item = np.load(self.data_path + self.data[index]).astype(int)
        #the input consists of the 2 grids in the last row and the present rules.
        #the label is the 9-th panel.
        return torch.from_numpy(np.concatenate((item[6*336:8*336], item[-29:]))), torch.from_numpy(item[8*336:-29])

    def __len__(self):
        return self.length

class GraphPGM(InMemoryDataset):

    def __init__(self, root, data_dir, transform=None, pre_transform=None):
        self.data_dir = data_dir
        super(GraphPGM, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        assert os.path.isdir(self.data_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []

        edge_index = torch.empty(2, 56, dtype=torch.long)
        c = 0
        for i in range(8):
            for j in range(8):
                if i != j:
                    edge_index[:, c] = torch.tensor([i, j], dtype=torch.long)
                    c += 1

        pos = torch.tensor([[0, 0],
                            [0, 1],
                            [0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2],
                            [2, 0],
                            [2, 1]])

        #Since I don't initialize data.pos, the GNN does not know the positions of the panels ==> invariance! (might also lead to poor performance)
        for filename in tqdm(os.listdir(self.data_dir)):
            RPM = np.load(self.data_dir + filename)
            x = torch.from_numpy(RPM[:8*336].reshape(8, 336))
            y = torch.from_numpy(RPM[-29:].reshape(1, 29))
            data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def rule_metrics(x, x_rules, device):
    warnings.filterwarnings('error')

    #x.to(device)
    #x_rules.to(device)

    #first, convert logits to hard assignments.
    y = x.view(-1, 3, 3, 345).float()
    existence = y[:,2,2,-9:]
    existence[existence > 0.5] = 1
    existence[existence <= 0.5] = 0
    existence = existence.int()
    y = y[:,:,:,:336]

    y /= 0.5
    y += 0.1
    sizes_oh = y[:, :, :, :99].view(-1, 3, 3, 9, 11) 
    cols_oh = y[:, :, :, 99:198].view(-1, 3, 3, 9, 11)
    types_oh = y[:, :, :, 198:270].view(-1, 3, 3, 9, 8)
    lcols_oh = y[:, :, :, 270:336].view(-1, 3, 3, 6, 11)

    sizes_oh[:,2,2] = F.one_hot((1 + sizes_oh[:,2,2,:,1:].argmax(2))*existence, num_classes=11) #9,11
    cols_oh[:,2,2] = F.one_hot((1+ cols_oh[:,2,2,:,1:].argmax(2))*existence, num_classes=11) #9,11
    types_oh[:,2,2] = F.one_hot((1+types_oh[:,2,2,:,1:].argmax(2))*existence, num_classes=8) #9,8
    lcols_oh[:,2,2] = F.one_hot(lcols_oh[:,2,2,:,1:].argmax(2), num_classes=11)

    sizes_oh = sizes_oh.long()
    cols_oh = cols_oh.long()
    types_oh = types_oh.long()
    lcols_oh = lcols_oh.long()

    #then, compute rules present in the hard assignments
    #compute attribute sets in one-hot encoding: result should have shape (64,3,3,4,10)
    attr_sets = torch.Tensor(64, 3, 3, 6, 10)
    panel_positions = torch.Tensor(64, 3, 3, 2, 10)

    
    #compute also for cols_oh and types_oh, check if all 3 are equal. If not, matrix is invalid!
    sz = torch.cat((torch.argmax(sizes_oh[:,2,2], dim=2), torch.zeros((64,1), dtype=torch.long, device=device)), 1) #(64,10)
    cl = torch.cat((torch.argmax(cols_oh[:,2,2], dim=2), torch.zeros((64,1), dtype=torch.long, device=device)), 1)
    ty = torch.cat((torch.argmax(types_oh[:,2,2], dim=2), torch.zeros((64,1), dtype=torch.long, device=device)), 1)
    sz[sz.nonzero(as_tuple=True)] = 1 
    cl[cl.nonzero(as_tuple=True)] = 1
    ty[ty.nonzero(as_tuple=True)] = 1
    
    valid = torch.prod(torch.logical_and(torch.eq(sz, cl), torch.eq(cl, ty)), dim=1) #(64)
    
    #valid = torch.eq(valid_aux, 90*torch.ones(64, device=device))

    panel_positions[:, :, :, 0] = torch.cat((torch.argmax(sizes_oh, dim=4), torch.zeros((64,3,3,1), dtype=torch.long, device=device)), axis=3) #(64,3,3,10)
    
    panel_positions[:, :, :, 1] = torch.cat((torch.argmax(lcols_oh, dim=4), torch.zeros((64,3,3,4), dtype=torch.long, device=device)), axis=3) #(64,3,3,10)

    attr_sets[:,:,:,0] = torch.sum(sizes_oh, dim=3)[:,:,:,1:]
    attr_sets[:,:,:,1] = torch.sum(cols_oh, dim=3)[:,:,:,1:]
    attr_sets[:,:,:,2] = torch.cat((torch.sum(types_oh, dim=3)[:,:,:,1:], torch.zeros((64,3,3,3), dtype=torch.long, device=device)), axis=3)
    attr_sets[:,:,:,3] = torch.sum(lcols_oh, dim=3)[:,:,:,1:]
    attr_sets[:,:,:,4:] = panel_positions

    attr_sets[attr_sets.nonzero(as_tuple=True)] = 1 #set all non-zero elements to 1.
    attr_sets = attr_sets.long()

    #compute rules matrix: result should have shape (64,29)
    rules = torch.Tensor(64,29)

    #XOR, OR, AND
    cardinalities = torch.sum(attr_sets, dim=4) #(64,3,3,6)
    card_flags = torch.prod(torch.prod(cardinalities, dim=1), dim=1) #(64,6) #True if set is non-empty
    third_sets = attr_sets[:,:,2] #(64,3,6,10)
    xors = torch.logical_xor(attr_sets[:,:,0].bool(), attr_sets[:,:,1].bool()).long() #(64,3,6,10)
    ors = torch.logical_or(attr_sets[:,:,0].bool(), attr_sets[:,:,1].bool()).long() #(64,3,6,10)
    ands = torch.logical_and(attr_sets[:,:,0].bool(), attr_sets[:,:,1].bool()).long() #(64,3,6,10)

    #check if constant
    full_ors = torch.logical_or(attr_sets[:,:,2].bool(), ors.bool()).long() #(64,3,6,10)
    full_ors_card_flags_1 = torch.prod(torch.eq(torch.sum(full_ors, dim=3), 1).long(), dim=1) #(64,6) checks if the full_ors have 1 element => implies constant rule
    card_flags_1 = torch.eq(cardinalities, 1).long() #(64,3,3,6)
    card_flags_1 = torch.prod(torch.prod(card_flags_1, dim=2), dim=1) #(64,6) #if all sets have 1 element
    not_constant = 1 - torch.mul(card_flags_1, full_ors_card_flags_1) #(64,6) ==1 if the value sets are not constant
    eq_aux_1 = torch.prod(torch.prod(torch.eq(attr_sets[:,:,0,4:], attr_sets[:,:,1,4:]).long(), dim=3), dim=1) #(64,2)
    eq_aux_2 = torch.prod(torch.prod(torch.eq(attr_sets[:,:,1,4:], attr_sets[:,:,2,4:]).long(), dim=3), dim=1) #(64,2)
    not_constant[:,4:] = 1 - torch.mul(eq_aux_1, eq_aux_2) #constraint slightly different for position rules
    
    ##can be done with one eq call, one mul call, and 2 prod calls in total, optimize!
    raw_xor_results = torch.eq(third_sets, xors).long() #(64,3,6,10)
    raw_or_results = torch.eq(third_sets, ors).long() #(64,3,6,10)
    raw_and_results = torch.eq(third_sets, ands).long() #(64,3,6,10)
    xor_results = torch.prod(torch.prod(raw_xor_results, dim=3), dim=1) #(64,6)
    or_results = torch.prod(torch.prod(raw_or_results, dim=3), dim=1) #(64,6)
    and_results = torch.prod(torch.prod(raw_and_results, dim=3), dim=1) #(64,6)
    xor_r = torch.mul(torch.mul(card_flags, xor_results).bool().long(), not_constant) #(64,6)
    or_r = torch.mul(torch.mul(card_flags, or_results).bool().long(), not_constant) #(64,6)
    and_r = torch.mul(torch.mul(card_flags, and_results).bool().long(), not_constant) #(64,6)
    
    #PROGR, UNION
    full_ors_card_flags = torch.prod(torch.eq(torch.sum(full_ors, dim=3), 3).long(), dim=1) #(64,6) checks if the full_ors have 3 elements
    identical_order_2 = torch.logical_and(torch.eq(attr_sets[:,0], attr_sets[:,1]), torch.eq(attr_sets[:,1], attr_sets[:,2])).long() #(64,3,6,10)
    non_identical_order_2 = 1 - torch.prod(torch.prod(identical_order_2, dim=3), dim=1) #(64,6)
    union_flag = torch.mul(torch.eq(full_ors[:,0], full_ors[:,1]).long(), torch.eq(full_ors[:,1], full_ors[:,2]).long()) #(64,6,10) checks if the 3 unions are consistent
    union_flag = torch.prod(union_flag, dim=2) #(64,6)
    union_flag *= non_identical_order_2
    progr_flag = 1 - union_flag # if the unions are inconsistent, card_flags * full_ors_card_flags * (NOT union_flag) == 1 implies progression.
                                # since we only have non-distractive RPMs, it MUST be the progression relation.
    aux = torch.mul(card_flags_1, full_ors_card_flags) #intermediate product to save computation
    union_r = torch.mul(aux, union_flag) #(64,6)
    progr_r = torch.mul(aux, progr_flag) #(64,6)

    #NUMBER PROGR, NUMBER UNION
    num_sets = cardinalities[:,:,:,4] #(64,3,3)
    num_avg = torch.true_divide(torch.sum(num_sets,dim=(1,2)), 9)
    non_const = 1 - torch.prod(torch.prod(torch.eq(num_sets - num_avg.view(-1,1,1), 0).long(), dim=2), dim=1)
    #nonzero = torch.prod(torch.prod(num_sets, dim=2), dim=1).bool().long()
    non_identical_order = 1 - torch.prod(torch.logical_and(torch.eq(num_sets[:,0], num_sets[:,1]), torch.eq(num_sets[:,1], num_sets[:,2])).long(), dim=1) #(64,3)
    nprog_r = torch.prod(torch.logical_and(torch.lt(num_sets[:,:,0], num_sets[:,:,1]), torch.lt(num_sets[:,:,1], num_sets[:,:,2])).long(), dim=1)
    nprog_r = torch.mul(nprog_r, non_const)
    nunion_r = torch.logical_and(torch.eq(torch.sort(num_sets[:,0])[0], torch.sort(num_sets[:,1])[0]), torch.eq(torch.sort(num_sets[:,1])[0], torch.sort(num_sets[:,2])[0])) #(64,3)
    nunion_r = torch.mul(torch.mul(torch.prod(nunion_r.long(), dim=1), non_const), non_identical_order)
    #i += 1
    
    #Reorder computed rules to allow direct comparison (use the rules Tensor)
    rules[:,:6] = xor_r
    rules[:,6:12] = or_r
    rules[:,12:18] = and_r
    rules[:,18] = nprog_r
    rules[:,19] = nunion_r
    rules[:,20:24] = progr_r[:,:4]
    rules[:,24:28]= union_r[:,:4]
    rules[:,28] = union_r[:,5]
    rule_mapping = np.array([20,0,6,12,24,21,1,7,13,25,18,19,4,10,16,22,2,8,14,26,23,3,9,15,27,5,11,17,28])
    rules = rules[:,rule_mapping].long()
    
    position_rule_flag = torch.sum(rules[:,12:15], dim=1).long() #(64), 1 iff a shape position rule exists
    rules[:,10:12] *= (1 - position_rule_flag).view(-1,1) #number rules can't exist if a position rule exists
    
    correct_vec = torch.prod(torch.eq(rules, x_rules).int(), dim=1)
    correct = torch.sum(correct_vec)
    total = 64
    
    moderate_correct_vec = torch.mul(valid, correct_vec.to(device))
    
    moderate_correct = torch.sum(moderate_correct_vec).float()
    
    flags = torch.zeros(64,6).to(device)
    
    flags[0] = torch.sum(rules[:,:5])   #shape.size
    flags[1] = torch.sum(rules[:,5:10]) #shape.color
    flags[2] = torch.sum(rules[:,10:15])#shape.num/pos
    flags[3] = torch.sum(rules[:,15:20])#shape.type
    flags[4] = torch.sum(rules[:,20:25])#line.color
    flags[5] = torch.sum(rules[:,25:29])   #line.type

    constant = torch.zeros(64,6).to(device)

    constant[:,0] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,0], attr_sets[:,0,0,0]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,0], dim=1), torch.ones(64)))
    constant[:,1] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,1], attr_sets[:,0,0,1]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,1], dim=1), torch.ones(64)))
    constant[:,2] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,4], attr_sets[:,0,0,4]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,4], dim=1), torch.ones(64)))
    constant[:,3] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,2], attr_sets[:,0,0,2]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,2], dim=1), torch.ones(64)))
    constant[:,4] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,3], attr_sets[:,0,0,3]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,3], dim=1), torch.ones(64)))
    constant[:,5] = torch.mul(torch.prod(torch.eq(attr_sets[:,2,2,5], attr_sets[:,0,0,5]), dim=1), torch.eq(torch.sum(attr_sets[:,2,2,5], dim=1), torch.ones(64)))

    constantness = torch.prod((flags + constant), dim=1)
    constantness[constantness > 0] = 1

    hard_correct_vec = torch.mul(constantness, correct_vec.to(device))
    hard_correct = torch.sum(hard_correct_vec).float()

    return correct, hard_correct, total, [sizes_oh, cols_oh, types_oh, lcols_oh]