#some other util functions.

import copy

import numpy as np
import pandas as pd
import operator
from collections import Counter
import scipy

import tensorflow as tf

def cal_sims(test_pair, feature, right=None):
    if right is None:
        feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
        feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
    else:
        feature_a = feature[:right]
        feature_b = feature[right:]
    fb = tf.transpose(feature_b, [1, 0])
    return tf.matmul(feature_a, fb)


# Hungarian algorithm, only for the CPU
# result = optimize.linear_sum_assignment(sims, maximize=True)
# test(result, "hungarian")

# Sinkhorn operation
def sinkhorn(sims, eps=1e-6):
    sims = tf.exp(sims * 50)
    for k in range(10):
        sims = sims / (tf.reduce_sum(sims, axis=1, keepdims=True) + eps)
        sims = sims / (tf.reduce_sum(sims, axis=0, keepdims=True) + eps)
    return sims

def index_tem_triples(triples, pair_dic):
    """
    index the temporal triples for yago and wiki set
    for the simplicity, we generate the index for (head, tail) pair.
    """
    rec = {}
    rec_w = {}
    count = 0
    for triple in triples:
        head, rel, tail, ts, te = triple    
        if ts != 0 or te != 0:
            head_w, tail_w = pair_dic[head], pair_dic[tail]
            if (head_w, tail_w) not in rec_w:
                rec_w[(head_w, tail_w)] = [count]
                rec_w[(head_w, tail_w)].append([head_w, rel, tail_w, ts, te])
            else:
                rec_w[(head_w, tail_w)].append([head_w, rel, tail_w, ts, te])

            if (head, tail) not in rec:
                rec[(head, tail)] = [count]
                rec[(head, tail)].append(triple)
                count += 1
            else:
                rec[(head, tail)].append(triple)
    return rec, rec_w

def index_tem_triples_w(triples):
    """
    index the temporal triples for yago and wiki set
    for the simplicity, we generate the index for (head, tail) pair.
    """
    rec = {}
    rec_w = {}
    for triple in triples:
        head, rel, tail, ts, te = triple
        #head_y, tail_y = all_pair_dic_inverse[head], all_pair_dic_inverse[tail]
        if ts != 0 or te != 0:
            if (head, tail) not in rec_w:
                rec_w[(head, tail)] = []
                rec_w[(head, tail)].append(triple)
            else:
                rec_w[(head, tail)].append(triple)
            #if (head_y, tail_y) not in rec:
            #    rec[(head_y, tail_y)] = []
            #    rec[(head_y, tail_y)].append([head_y, rel, tail_y, ts, te])
            #else:
            #    rec[(head_y, tail_y)].append([head_y, rel, tail_y, ts, te])
    return rec_w

def check_align_pair(rec_tem_1, rec_tem_2):
    """check the aligned pairs for the two KGs"""
    rec = {}
    #iterate the yago part
    for pairs, info_y in rec_tem_1.items():
        head, tail = pairs
        if pairs in rec_tem_2:
            rec[pairs] = [info_y, rec_tem_2[pairs]]
        elif (tail, head) in rec_tem_2:
            rec[pairs] = [info_y, rec_tem_2[(tail, head)]]
    return rec

def extract_temp_fact(rec_align):
    """visualize with the temporal info"""
    rec = {}
    for pairs, infos in rec_align.items():
        info_y = infos[0][1:]
        info_w = infos[1]
        info_y_ = []
        for i in info_y:
            ts, te = i[-2], i[-1]
            info_y_.append([ts, te])
        info_w_ = []
        for i in info_w:
            ts, te = i[-2], i[-1]
            info_w_.append([ts, te])
        rec[pairs] = [info_y_, info_w_]
    return rec

def process_temp(rec_temp):
    """filter the missing temporal with the existing value"""
    rec = {}
    for pair, infos in rec_temp.items():
        info_y = infos[0]
        info_w = infos[1]
        info_y_, info_w_ = [], []
        for info in info_y:
            ts, te = info
            #if ts == 0:
            #    print(1)
            if te == 0:
                te = ts
            info_y_.append([ts, te])
        for info in info_w:
            ts, te = info
            #if ts == 0:
            #    print(ts, te)
            if te == 0:
                te = ts
            if ts == 0:
                ts = te
            info_w_.append([ts, te])
        rec[pair] = [info_y_, info_w_]
    return rec

def gen_predict_dic(ranks_index, ranks_dic_wiki, all_pair_dic):
    """get the predict dict based on the ranks index"""
    rec = {}
    for index_y, index_w in all_pair_dic.items():
        if index_y not in ranks_index:
            rec[index_y] = index_w
        else:
            rec[index_y] = ranks_dic_wiki[index_y][0]
    return rec

def extract_hetero_temp(rec_temp_p):
    """extract the heterogenous temeporal for the temporal triples"""
    rec = {}
    rec_homo = {}
    for pair, infos in rec_temp_p.items():
        sym = 0
        info_y = infos[0]
        info_w = infos[1]
        for info in info_y:
            if info in info_w:
                sym = 1
        if sym == 1:
            rec_homo[pair] = infos
        else:
            rec[pair] = infos
    return rec, rec_homo

def gen_hetero_set(hetero_info):
    rec = {}
    for pair, infos in hetero_info.items():
        info_y, info_w = infos[0], infos[1]
        rec[pair] = set()
        for i in info_y:
            for j in info_w:
                rec[pair].add(str([i, j]))
    return rec

def hetero_refine(hetero_set):
    """refine the hetero set with overlapping temporal only"""
    rec = {}
    for pair, intervals in hetero_set.items():
        for interval in intervals:
            interval = eval(interval)
            tem_1, tem_2 = interval[0], interval[1]
            ts_1, te_1 = tem_1[0], tem_1[1]
            ts_2, te_2 = tem_2[0], tem_2[1]
            #check the overlapping
            if ts_1 <= te_2 and ts_2 <= te_1:
                if pair not in rec:
                    rec[pair] = set()
                    rec[pair].add(str(interval))
                else:
                    rec[pair].add(str(interval))
    return rec

def gen_interval_pair(hetero_set):
    """generate {interval_pair: ent}"""
    rec = {}
    for pair, hetero_temps in hetero_set.items():
        for hetero_temp in hetero_temps:
            hetero_temp_ = eval(hetero_temp)
            hetero_temp_i = str([hetero_temp_[1], hetero_temp_[0]])
            if hetero_temp not in rec:
                rec[hetero_temp] = set()
                rec[hetero_temp].add(pair)
            else:
                rec[hetero_temp].add(pair)
            if hetero_temp_i not in rec:
                rec[hetero_temp_i] = set()
                rec[hetero_temp_i].add(pair)
            else:
                rec[hetero_temp_i].add(pair)
    return rec

def expand_set(target_set):
    rec = set()
    for tuple in target_set:
        for i in tuple:
            rec.add(i)
    return rec

def gen_predict_inverse(predict_dic):
    """generate the predict inverse verion {wiki_index: set(yago_index)}"""
    rec = {}
    for yago_index, wiki_index in predict_dic.items():
        if wiki_index not in rec:
            rec[wiki_index] = set()
            rec[wiki_index].add(yago_index)
        else:
            rec[wiki_index].add(yago_index)
    return rec

def gen_counter_set(target_set, predict_dic_set_i):
    rec = set()
    for ent in target_set:
        add_set = predict_dic_set_i[ent]
        for i in add_set:
            rec.add(i)
    return rec

def update_ent_int(hetero_set_dict, ent_tem_int_time_count, predict_dic_set_i):
    """update the ent_tem_int_time_count based on the hetero_set_dict"""
    ent_int_dict = copy.deepcopy(ent_tem_int_time_count)
    for hetero_int, pairs in hetero_set_dict.items(): 
        hetero_int = eval(hetero_int)
        hetero_list = [tuple(hetero_int[0]), tuple(hetero_int[1])]
        wiki_add_set = expand_set(pairs)
        yago_add_set = gen_counter_set(wiki_add_set, predict_dic_set_i)
        for ent in wiki_add_set:
            for hetero_temp in hetero_list:
                if hetero_temp not in ent_int_dict[ent]:
                    ent_int_dict[ent][hetero_temp] = 1
                else: 
                    ent_int_dict[ent][hetero_temp] += 1

        for ent in yago_add_set:
            for hetero_temp in hetero_list:
                if hetero_temp not in ent_int_dict[ent]:
                    ent_int_dict[ent][hetero_temp] = 1
                else: 
                    ent_int_dict[ent][hetero_temp] += 1

    return ent_int_dict

def gen_valid_set(ent_int_dict, all_pair_dic):
    rec_y = set()
    rec_w = set()
    for ent in ent_int_dict:
        if ent in all_pair_dic:
            for i in ent_int_dict[ent]:
                rec_y.add(i)
        else:
            for i in ent_int_dict[ent]:
                rec_w.add(i)
    return rec_y & rec_w

def load_triples_(file_name):
    """load the  triples for the specific file"""
    triples = []
    for line in open(file_name, 'r'):
        params = line.split()
        head = params[0]
        rel = params[1]
        tail = params[2]
        ts = params[3]
        te = params[4]
        triples.append([head, rel, tail, ts, te])
    return triples

def convert_triple(triples):
    """convert the triple representation from string to int"""
    rec = []
    for triple in triples:
        head, rel, tail, ts, te = int(triple[0]), \
        int(triple[1]), int(triple[2]), int(triple[3]), int(triple[4])
        rec.append([head, rel, tail, ts, te])
    return rec

def convert_index(index_wrong, ranks, dev_pair):
    """
    convert the wrong_index, ranks to the index in the respective KGs
    Note that dev_pair[:, 0] is the index of the subjective KG
    the dev_pair[:, 1] is the index of the target KG. 
    """
    new_indexes = []
    new_ranks_yago = []
    new_ranks_wiki = []
    for index in index_wrong:
        #get the wrong index in the subjective KG
        new_index = list(dev_pair[index][:, 0])
        new_indexes.append(new_index)
    for rank_list in ranks:
        rank_yago = []
        rank_wiki = []
        for rank in rank_list:
            #get the wrong prediction for the entity to be predicted
            #the result is the target KG, with the prediction from high to low
            new_rank_yago = dev_pair[rank][:, 0]
            new_rank_wiki = dev_pair[rank][:, 1]
            rank_yago.append(new_rank_yago)
            rank_wiki.append(new_rank_wiki)
        new_ranks_yago.append(rank_yago)
        new_ranks_wiki.append(rank_wiki)
    return new_indexes, new_ranks_yago, new_ranks_wiki

def build_dic(new_index_wrong, new_ranks_yago, new_ranks_wiki, wrong_rank, new_scores, dev_pair_dic):
    """
    build the dict for the {index_wrong: the right counterpart} and the dict {index_wrong: the actual ranking}
    For this, we could get the predicted entity index for the target KG, and the true label for the wrong 
    predicted entity.
    """
    #get the {index: the target label}
    index_wrong_dic = []
    for index_list in new_index_wrong:
        wrong_dic = {}
        for element in index_list:
            wrong_dic[element] = dev_pair_dic[element]
        index_wrong_dic.append(wrong_dic)
    #get the [dict {index: rank}]
    ranks_index = []
    for i, index_list in enumerate(new_index_wrong):
        ranks_index.append(dict(zip(index_list, wrong_rank[i])))

    #get the [dict {index: the yago predicted array}]
    ranks_dic_yago = []
    for i, index_list in enumerate(new_index_wrong):
        ranks_dic_yago.append(dict(zip(index_list, new_ranks_yago[i])))
    
    #get the [dict {index: the wiki predicted array}]
    ranks_dic_wiki = []
    for i, index_list in enumerate(new_index_wrong):
        ranks_dic_wiki.append(dict(zip(index_list, new_ranks_wiki[i])))
    
    #get the [dict {index: score array}]
    scores_dic = []
    for i, index_list in enumerate(new_index_wrong):
        scores_dic.append(dict(zip(index_list, new_scores[i])))
    
    return index_wrong_dic, ranks_index, ranks_dic_yago, ranks_dic_wiki, scores_dic

