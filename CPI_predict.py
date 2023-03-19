# changed
import os
#
import sys
import math
import time
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from pdbbind_utils import *
from CPI_model import *


def test(test_data, params,batch_size, state_dict):
    init_A, init_B, init_W = loading_emb(measure)
    net = Net(init_A, init_B, init_W, params)
    # net.cuda()

    net.load_state_dict(torch.load(state_dict,map_location = "cpu")) #
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print 'total num params', pytorch_total_params

    output_list = []
    label_list = []
    pairwise_auc_list = []
    for i in range(int(math.ceil(len(test_data[0]) / float(batch_size)))):
        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, aff_label, pairwise_mask, pairwise_label = \
            [test_data[data_idx][i * batch_size:(i + 1) * batch_size] for data_idx in range(9)]

        inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
        affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

        output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += aff_label.reshape(-1).tolist()
    output_list = np.array(output_list)
    return output_list


if __name__ == "__main__":
    # result_path
    result_dict = {}
    result_df_path = "./predict_result_home/"
    if not os.path.exists(result_df_path):
        os.makedirs(result_df_path)

    # load data
    # with open('../preprocessing/pdbbind_all_combined_input_' + measure, 'rb') as f:
    data_pack_path = ""  # todo
    with open(data_pack_path, 'rb') as f:
        data_pack = pickle.load(f)
    ligand_ids = data_pack[7]
    test_idx = [i for i in range(len(ligand_ids))]
    print 'test num:', len(ligand_ids)
    test_data = data_from_index(data_pack, test_idx)
    result_dict["ligand_ids"] = ligand_ids

    # load model
    saved_net_path = "./saved_model_home/"
    saved_models = os.listdir(saved_net_path)
    for sub_model in saved_models:
        sub_model_file = saved_net_path + sub_model
        measure, setting, clu_thre, a_rep, a_fold = sub_model.split(".")[0].split("_")[-5:]
        model_tag = "_".join(str(measure), str(clu_thre), str(a_rep), str(a_fold))

        '''
        if not os.path.exists(saved_net_path):
            os.makedirs(saved_net_path)
        
        #
        #evaluate scheme
        
        measure = sys.argv[1]  # IC50 or KIKD
        setting = sys.argv[2]   # new_compound, new_protein or new_new
        clu_thre = float(sys.argv[3])  # 0.3, 0.4, 0.5 or 0.6
        n_epoch = 30
        n_rep = 10
        
        assert setting in ['new_compound', 'new_protein', 'new_new']
        assert clu_thre in [0.3, 0.4, 0.5, 0.6]
        assert measure in ['IC50', 'KIKD']
        '''
        GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
        if setting == 'new_compound':
            n_fold = 5
            batch_size = 32
            k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 128, 128
        elif setting == 'new_protein':
            n_fold = 5
            batch_size = 32
            k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128
        elif setting == 'new_new':
            n_fold = 9
            batch_size = 32
        # k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
        para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']

        params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]
        # params = sys.argv[4].split(',')
        # params = map(int, params)

        # print evaluation scheme
        print 'Dataset: PDBbind v2018 with measurement', measure
        print 'Clustering threshold:', clu_thre
        print 'Number of epochs:', n_epoch
        print 'Number of repeats:', n_rep
        print 'Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)]

        test_output = test(test_data, params, batch_size, sub_model_file)

        result_dict[model_tag] = test_output
    # save result
    df_res = pd.DataFrame.from_dict(result_dict)
    pd.to_csv(result_df_path)
    print("done")
