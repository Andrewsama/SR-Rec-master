import json
import numpy as np
import random
import sys

# {商品id:关联buy商品id的list}，商品数，训练负样本数2，测试负样本数100
def pair_construct(item_dict, num_items, train_neg_num, test_neg_num, random_seed = 1):

    random.seed(random_seed)
    all_items = np.arange(num_items)
    com_edge_index = []
    train_triple_pair = []
    val_triple_pair = []
    test_triple_pair = []

    for key_item in item_dict.keys():
        pos_list = item_dict[key_item]      # 关联buy商品id的list
        neg_list = list(set(all_items)-set(pos_list))   # 不关联buy商品id的list (负样本)
        # 训练中每个商品对应的每个正样本采两个负样本，验证和测试中每个商品都采集100个负样本
        neg_sample = random.sample(neg_list, train_neg_num * len(pos_list) + 2 * test_neg_num)
        if len(pos_list) < 3:
            #if the length of pos_list is smaller than 3, do not put it into the validation set and test set
            for i in range(len(pos_list)):
                tem = [int(key_item), int(pos_list[i])]
                com_edge_index.append([int(key_item), int(pos_list[i])])
                tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i + 1)])
                train_triple_pair.append(tem)
            continue

        for i in range(len(pos_list)-2):
            tem = [int(key_item), int(pos_list[i])]
            com_edge_index.append([int(key_item), int(pos_list[i])])
            tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i+1)])
            train_triple_pair.append(tem)

        tem = [ int(key_item), int(pos_list[len(pos_list)-2])]
        tem.extend(neg_sample[train_neg_num * len(pos_list):  train_neg_num * len(pos_list) + test_neg_num])
        val_triple_pair.append(tem)

        tem = [ int(key_item), int(pos_list[len(pos_list)-1])]
        tem.extend(neg_sample[train_neg_num * len(pos_list) + test_neg_num: ])
        test_triple_pair.append(tem)

    train_triple_pair = np.array(train_triple_pair)
    val_triple_pair = np.array(val_triple_pair)
    test_triple_pair = np.array(test_triple_pair)
    # com_edge_index: [[商品id1，关联的buy商品id1],[商品id1，关联的buy商品id2],...]
    # train_triple_pair: [[商品id1，关联的buy商品id1，非关联的buy商品id3，非关联的buy商品id4]，[商品id1，关联的buy商品id2，非关联的buy商品id5，非关联的buy商品id6]]
    # val_triple_pair: [[商品id1，关联的buy商品id1，非关联的buy商品id3,...,非关联的buy商品id102]，[商品id1，关联的buy商品id2，非关联的buy商品id5,...,非关联的buy商品id104]]
    # test_triple_pair和val_triple_pair的格式一致，不同的是对负样本的选择，且两者都不将关联buy样本数少于3的商品放入(但train中是放入的)
    return com_edge_index, train_triple_pair, val_triple_pair, test_triple_pair

def main():
    print("Generating training, validation and test sets...")

    com_dict = {}
    features = []

    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            js = json.loads(line)
            neighbors = js['neighbor']
            key_item = js['node_id']
            com_list = list(neighbors["1"].keys())
            com_dict[key_item] = com_list   # {商品id:关联buy商品id的list}
            features.append(list(js['uint64_feature'].values())) # [标签1下标，标签2下标，float价格]

    features = np.squeeze(np.array(features))
    num_items = features.shape[0]
    # {商品id:关联buy商品id的list}，商品数，训练负样本数2，测试负样本数100
    train_com_edge_index, train_set, val_set, test_set = pair_construct(com_dict, num_items, train_neg_num, test_neg_num)

    np.savez(save_path, features = features, com_edge_index = train_com_edge_index, train_set = train_set,
             val_set = val_set, test_set = test_set)

    print("train set num: {}; validation set num: {}; test set num: {}".
          format(len(train_set), len(val_set), len(test_set)))

if __name__ == '__main__':
    data_name = sys.argv[1]

    dataset_path = "./data/{}.json".format(data_name)
    save_path = "./processed/{}.npz".format(data_name)

    train_neg_num = 2
    test_neg_num = 100
    main()









