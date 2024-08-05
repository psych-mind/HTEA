#this script, we test with the other method to intilize with the random seeds.
import numpy as np
from utils import *
from models import *
from tqdm import *
import torch
import torch.nn.functional as F
import keras.backend as KTF

import pandas as pd
import dto
import time
from utils_side import *
from args import args

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices(
    device_type="GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

#load the parameters.
#eval_epoch = 3

node_hidden = args.dim
rel_hidden = node_hidden
time_hidden = int(node_hidden / 2)
batch_size = args.batchsize
dropout_rate = args.dropout
lr = args.lr
gamma = args.gamma
depth = args.depth
device = 'cuda:0'
train_ratio = args.seed

seed = args.randomseed
torch.manual_seed(seed)

seed = 12345
np.random.seed(seed)

file_path = '/scratch/user/uqjli48/DualMatch/'
filename = 'data/data_h/'

ts_ = time.time()
train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
 time_features, time_features_int, radj, time_int_dict_i, ent_int_dict = load_data(
    file_path + filename, train_ratio=train_ratio)

ref_pair = np.concatenate((train_pair, dev_pair))
dev_pair_dic = dict(zip(dev_pair[:, 0], dev_pair[:, 1]))
all_pair_dic = dict(zip(ref_pair[:, 0], ref_pair[:, 1]))
all_pair_dic_inverse = dict(zip(ref_pair[:, 1], ref_pair[:, 0]))

#load_data(lang, train_ratio=0.3, ent_int_dict=None, valid_int_set=None, unsup=False)

adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
time_int_matrix, time_int_val = np.stack(time_features_int.nonzero(), axis=1), \
time_features_int.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
time_size = time_features.shape[1]
time_int_size = time_features_int.shape[1]
ent_size = 0

triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)

#training_time = 0.
#grid_search_time = 0.
#time_encode_time = 0.

def get_embedding(index_a, index_b, vec):
    vec = vec.detach().numpy()
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec

def save_suffix(save_final=False, cnt_call=1,\
 sym='_hetero_try_args'):
    suf = '_' + str(len(train_pair))
    suf += sym
    return suf

def align_loss(align_input, embedding, node_size):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        # may not work
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))

    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]
    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    r_neg_dis = squared_dist([r_emb, embedding])
    l_neg_dis = squared_dist([l_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) \
    - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) \
    - F.one_hot(right, num_classes=node_size)).to(device)
    # modified
    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std = torch.std(r_loss, dim=-1, keepdim=True)
        r_loss.data = (r_loss.data - r_mean) / r_std
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std = torch.std(l_loss, dim=-1, keepdim=True)
        l_loss.data = (l_loss.data - l_mean) / l_std

    lamb, tau = 30, 10
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)

#here we raise two problems:
#we retrain the model or to continue training the current trained modle.
#we may try it in two branches.

def train(time_int_matrix, time_int_val, sym, load=None):
    print('begin')
    model = OverAll(node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
                    rel_size=rel_size, rel_hidden=rel_hidden,
                    time_size=time_size, time_int_size=time_int_size,
                    rel_matrix=rel_matrix, ent_matrix=ent_matrix,
                    time_matrix=time_matrix, time_val=time_val,
                    time_int_matrix=time_int_matrix, time_int_val=time_int_val,
                    triple_size=triple_size, dropout_rate=dropout_rate,
                    depth=depth, device=device)
    model = model.to(device)
    if load is not None:
        model.load_state_dict(torch.load(load))
    # opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0)
    print('model constructed')

    #evaluater = evaluate(dev_pair)
    #rest_set_1 = [e1 for e1, e2 in dev_pair]
    #rest_set_2 = [e2 for e1, e2 in dev_pair]
    #np.random.shuffle(rest_set_1)
    #np.random.shuffle(rest_set_2)

    epoch = 10 if train_ratio > 0.2 else 20
    print(epoch)
#    if unsup:
#        epoch = 3
    
    start = time.time()
    for turn in range(1):
        tic = time.time()
        for i in trange(epoch):
            np.random.shuffle(train_pair)
            for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                          range(len(train_pair) // batch_size + 1)]:
                #for tem_pairs in [tem_seeds[i * batch_size:(i + 1) * batch_size] for i in
                #          range(len(tem_seeds) // batch_size + 1)]:
                inputs = [adj_matrix]
                output_ent = model(inputs)
                loss_ent = align_loss(pairs, output_ent, node_size)
                #loss_tem = tem_loss(tem_emb, tem_reg_array[:, 0], tem_reg_array[:, 1])
    
                #print(loss_ent, loss_tem)
                print(loss_ent)
                opt.zero_grad()
                loss_ent.backward(retain_graph=True)
                #loss_tem.backward()
                #loss_tem.backward()
                opt.step()
            if i == epoch - 1:
                toc = time.time()
                model.eval()
                with torch.no_grad():
                    output_ent = model(inputs)
                    output2 = output_ent.cpu().numpy()
                    output2 = output2 / (np.linalg.norm(output2, axis=-1, keepdims=True) + 1e-5)
                    dto.saveobj(output2, 'embedding_of_' + save_suffix() + '_' + str(sym))
                model.train()
        training_time = toc - tic
    return model

#print(save_suffix())
#model = train(time_int_matrix, time_int_val, sym='1')
#output = dto.readobj('embedding_of_' + save_suffix() + '_' + sym)

#output = tf.convert_to_tensor(output)
#sim = cal_sims(dev_pair, output)
#sim_s = sinkhorn(sim)

#detect the heterogentiy pair based on the result.
import sys
sys.path.append('/scratch/user/uqjli48/week_10_/dual_match_util/')
sys.path.append('/scratch/user/uqjli48/week_10_')
from CSLS_test_4s_ import *
from CSLS_test_4 import *
from CSLS_test_3 import *


def gen_hetero_stat(ranks_index, ranks_dic_wiki, all_pair_dic, ent_int_dict):
    """update the ent_int_dict and the valid_int_set"""
    #heterogeneity detection.
    predict_dic = gen_predict_dic(ranks_index[0], ranks_dic_wiki[0], all_pair_dic)
    new_triples_1 = load_triples_(file_path + filename + 'triples_1')
    new_triples_2 = load_triples_(file_path + filename + 'triples_2')
    new_triples_1_ = convert_triple(new_triples_1)
    new_triples_2_ = convert_triple(new_triples_2)
    #the predicted result.
    rec_tem_1, rec_tem_wr1 = index_tem_triples(new_triples_1_, predict_dic)
    rec_tem_w2 = index_tem_triples_w(new_triples_2_)
    rec_align_wr = check_align_pair(rec_tem_wr1, rec_tem_w2)
    rec_temp_wr = extract_temp_fact(rec_align_wr)
    rec_temp_wrp = process_temp(rec_temp_wr)

    hetero_info, homo_info = extract_hetero_temp(rec_temp_wrp)
    hetero_set = gen_hetero_set(hetero_info)
    hetero_set_refine = hetero_refine(hetero_set)
    hetero_set_dict_refine = gen_interval_pair(hetero_set_refine)

    predict_dic_set_i = gen_predict_inverse(predict_dic)
    #update the ent_int_dict
    ent_int_dict = update_ent_int(hetero_set_dict_refine, ent_int_dict, predict_dic_set_i)
    #update the valid_int_set
    #set_1 = set(np.concatenate((train_pair[:, 0], dev_pair[:, 0])))
    valid_int_set = gen_valid_set(ent_int_dict, all_pair_dic)
    return ent_int_dict, valid_int_set


#here we do the iterative heterogeneity training with customized parameters.
iter = args.iteration
sym_list = list(np.arange(iter))

def iter_train(iter, sym_list, time_int_matrix, time_int_val, \
dev_pair, all_pair_dic, ent_int_dict):
    """train the model with the customized iteration"""
    for i in range(iter):
        sym = int(sym_list[i])
        ts = time.time()
        #the first iteration
        if sym == 0:
            try:
                output = dto.readobj('embedding_of_' + save_suffix() + '_' + str(sym))
            except:
                model = train(time_int_matrix, time_int_val, sym)
                torch.save(model.state_dict(), 'embedding_of_' + \
                save_suffix() + '_' + str(sym) + '.pt')
                output = dto.readobj('embedding_of_' + save_suffix() + '_' + str(sym))
        else:
            ent_int_dict, valid_int_set = gen_hetero_stat(ranks_index, \
            ranks_dic_wiki, all_pair_dic, ent_int_dict)
            train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
            time_features, time_features_int, radj, time_int_dict_i, ent_int_dict \
             = load_data(file_path+filename, train_ratio, ent_int_dict, valid_int_set)
            time_int_matrix, time_int_val = np.stack(time_features_int.nonzero(), axis=1), \
            time_features_int.data
            load_path = 'embedding_of_' + save_suffix() + '_' + str(sym-1) + '.pt'
            try:
                output = dto.readobj('embedding_of_' + save_suffix() + '_' + str(sym))
            except:
                model = train(time_int_matrix, time_int_val, sym, load_path)
                torch.save(model.state_dict(), 'embedding_of_' + \
                save_suffix() + '_' + str(sym) + '.pt')
                output = dto.readobj('embedding_of_' + save_suffix() + '_' + str(sym))
        
        te = time.time()
        print(str(format((te - ts), '.2f')))
        output = tf.convert_to_tensor(output)
        sim = cal_sims(dev_pair, output)
        sim_s = sinkhorn(sim)
       
        #for each iteration we test the model performance
        print('iter result ' + str(i))
        print(multi_thread_cal_(sim_s.numpy(), 20, [1, 5, 10]))
     
        if sym != int(sym_list[-1]):
            hits, acc, mean, wrong_index, wrong_rank, wrong_rank_index, wrong_scores \
            = multi_thread_cal(sim_s.numpy(), 20, [1, 5, 10])
            new_indexes, new_ranks_yago, new_ranks_wiki = \
            convert_index(wrong_index, wrong_rank_index, dev_pair)
            index_wrong_dic, ranks_index, ranks_dic_yago, ranks_dic_wiki, scores_dic \
            = build_dic(new_indexes, new_ranks_yago, \
            new_ranks_wiki, wrong_rank, wrong_scores, dev_pair_dic)

    return sim_s

sim_s = iter_train(iter, sym_list, time_int_matrix, \
time_int_val, dev_pair, all_pair_dic, ent_int_dict)
te_ = time.time()
print('whole time consumption = ' +  str(format((te_ - ts_), '.2f')))

#the evaluation
#so here we set a threshold to calculate the time sensitive part of the entity.
def time_sensitive_split(tem_ratio, threshold=0.5):
    """split the tem_sensitive with a threshold split"""
    rec_n = set()
    rec_l = set()
    rec_h = set()
    for index, tem_raito_list in tem_ratio.items():
        yago_ratio, wiki_ratio = tem_raito_list
        if yago_ratio > threshold and wiki_ratio > threshold:
            rec_h.add(index)
        elif yago_ratio == 0 and wiki_ratio == 0:
            rec_n.add(index)
        else:
            rec_l.add(index)
    return rec_n, rec_l, rec_h

import pickle
    
with open(file_path + 'tem_ratio_dual.pkl', 'rb') as fp:
    ratio_dic = pickle.load(fp)
    print('good')

rec_n, rec_l, rec_h = time_sensitive_split(ratio_dic)
set_list_s = [rec_n, rec_l, rec_h]

dev_dic_i = dict(zip(np.arange(len(dev_pair)), np.array(dev_pair)[:, 0]))
dev_pair_dic = dict(zip([yago_index for yago_index, wiki_index in dev_pair],\
                        [wiki_index for yago_index, wiki_index in dev_pair]))

#exclude the entities not in the dec_pair_dic
def gen_exclude_dev(set_list):
    """exclude the dev entity in the set list"""
    set_list_r = []
    for set in set_list:
        set_extract = [i for i in set if i in dev_pair_dic]
        set_list_r.append(set_extract)
        print(len(set_extract))
    return set_list_r

set_list_s = gen_exclude_dev(set_list_s)
print('sensitive_test')
print('rel result sink')
print(multi_thread_cal_s_(sim_s.numpy(), 20, [1, 5, 10], set_list_s, dev_dic_i))

print('_________________________________')
print('heteogeneous test')

with open(file_path + 'set_hetero_split.pkl', 'rb') as fp:
    set_list_h = pickle.load(fp)
    print('good')
#print(len(set_list_h))
print([len(i) for i in set_list_h])

set_list_h = gen_exclude_dev(set_list_h)
print([len(i) for i in set_list_h])

print('rel result sink')
print(multi_thread_cal_s_(sim_s.numpy(), 20, [1, 5, 10], set_list_h, dev_dic_i))

print('_________________________________')
print('overall test')

print('rel result sink')
print(multi_thread_cal_(sim_s.numpy(), 20, [1, 5, 10]))


