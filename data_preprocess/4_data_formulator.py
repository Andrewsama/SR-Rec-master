import json
import sys


def assign_num_id(u, node_to_num_id): # 为每个用户重指定下标并返回
  if u not in node_to_num_id:
    node_to_num_id[u] = len(node_to_num_id)
  return node_to_num_id[u]


def init_node(num_id):
    return {
        "node_id": num_id,
        "node_type": 0,
        "node_weight": 1.0,
        "neighbor": {"0": [], "1": [], "2": [], "3": []},
        "uint64_feature": {},
        "edge":[]
    }


def add_neighbor(u, v, edge_type, node_data):
    node_data[u]['neighbor'][str(edge_type)].append(v)


def add_edge(u, v, edge_type, node_data):
  uv_edge = {
            "src_id": u,
            "dst_id": v,
            "edge_type": edge_type,
            "weight": 1.0,
  }
  node_data[u]['edge'].append(uv_edge)


def fill_node_features(node_data, node_to_num_id, valid_node_asin):
  def id_mapping(s, id_dict):
    if s not in id_dict:
        id_dict[s] = len(id_dict)
    return id_dict[s]

  cid1_dict, cid2_dict, cid3_dict = {}, {}, {}

  for eachline in meta_file:
    data = eval(eachline)
    asin = data['asin']
    if asin in valid_node_asin:
      c1, c2, c3 = data['category'][1:4]

      price = data['price']
      # 获得商品类别的id映射
      cid1, cid2, cid3 = id_mapping(c1, cid1_dict), id_mapping(c2, cid2_dict), \
                         id_mapping(c3, cid3_dict)

      num_id = node_to_num_id[asin]   # 获得商品的id映射

      node_data[num_id]['uint64_feature'] = {"0": [cid2], "1": [cid3], "2": [float(price[1:])]}

  for asin, num_id in cid2_dict.items():
    cid2_dict_file.write("{}\t{}\n".format(num_id, asin))
  for asin, num_id in cid3_dict.items():
    cid3_dict_file.write("{}\t{}\n".format(num_id, asin))

  feature_stats = "#cid2: {}; #cid3: {};".format(len(cid2_dict), len(cid3_dict))
  print(feature_stats)
  log_file.write(feature_stats + '\n')

def price_string(price):
  price = price[1:]
  return float(price)

def main():
  print("Converting node data to json format...")
  
  node_to_num_id = {}
  node_data = {}

  valid_node_asin = set()

  for eachline in sim_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    valid_node_asin.add(u)
    valid_node_asin.add(v)
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    if uid not in node_data:
      # neighbor中“0”，”1“表示连接关系的类别
      # uint64_feature表示商品的属性，为[id_类别1, id_类别2id, 浮点数_价格]
      node_data[uid] = init_node(uid) # 生成id对应字典  "node_id": num_id,"node_type": 0,"node_weight": 1.0,"neighbor": {"0": [], "1": [], "2": [], "3": []},"uint64_feature": {},"edge":[]
    if vid not in node_data:
      node_data[vid] = init_node(vid)

    add_neighbor(uid, vid, 0, node_data)   # node_data[u]['neighbor'][str(edge_type)].append(v)  # 将view下的边类别设为0
    add_neighbor(vid, uid, 0, node_data) 

  for eachline in cor_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    valid_node_asin.add(u)
    valid_node_asin.add(v)
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    if uid not in node_data:
      node_data[uid] = init_node(uid) 
    if vid not in node_data:
      node_data[vid] = init_node(vid)

    add_neighbor(uid, vid, 1, node_data)  # node_data[u]['neighbor'][str(edge_type)].append(v)  # 将buy下的边类别设为1
    add_neighbor(vid, uid, 1, node_data)

  fill_node_features(node_data, node_to_num_id, valid_node_asin)

  sim_num, cor_num = 0, 0

  for u in sorted(node_data.keys()):
    u_data = node_data[u]
    u_neighbor = node_data[u]['neighbor']

    sim_num += len(u_neighbor['0'])    # 获得商品的关联view数
    cor_num += len(u_neighbor['1'])    # 获得商品的关联buy数

    for edge_type in [0, 1]:
        for v in u_data['neighbor'][str(edge_type)]: # 将node_data中的'edge'赋值为"src_id": u,"dst_id": v,"edge_type": edge_type,"weight": 1.0,
            uid, vid = u, v
            add_edge(uid, vid, edge_type, node_data)
        u_data['neighbor'][str(edge_type)] = {
                v: 1.0 for v in u_data['neighbor'][str(edge_type)]}
    json.dump(u_data, out_file)
    out_file.write('\n')


  node_stats = "Total node num: {}".format(len(node_data)) 
  edge_stats = "Total edge num: {}; sim: {}; cor: {}".format(sim_num + cor_num, sim_num, cor_num)
  data_stats = "{}\n{}".format(node_stats, edge_stats)

  print(data_stats)
  log_file.write(data_stats)

  for asin, num_id in node_to_num_id.items():
    id_dict_file.write("{}\t{}\n".format(asin, num_id))



if __name__ == '__main__':
  data_name = sys.argv[1]

  sim_filename = "filtered_{}_sim.edges".format(data_name)
  cor_filename = "filtered_{}_cor.edges".format(data_name)

  sim_edges_file = open("./tmp/" + sim_filename, 'r') 
  cor_edges_file = open("./tmp/" + cor_filename, 'r') 

  meta_file = open('./tmp/filtered_meta_{}.json'.format(data_name)).readlines()

  log_file = open("./stats/{}.log".format(data_name), 'w')                    # 记录标签2类别数，标签3类别数，总节点数，总边数，view和buy各自边数
  id_dict_file = open('./tmp/{}_id_dict.txt'.format(data_name), 'w')          # 存储商品编号对应的下标
  cid2_dict_file = open('./tmp/{}_cid2_dict.txt'.format(data_name), 'w')      # 标签2对应下标
  cid3_dict_file = open('./tmp/{}_cid3_dict.txt'.format(data_name), 'w')      # 标签3对应下标
  out_file = open("./data/{}.json".format(data_name), 'w')                    # 存储包含整个数据集的node_data

  main()

