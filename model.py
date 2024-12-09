import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.metrics import ndcg_score

class Twostage_Attention(nn.Module):
    def __init__(self, hidden_size):    # 16
        super(Twostage_Attention, self).__init__()

        self.hidden_size = hidden_size

        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)

        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        self.nn_cat = nn.Linear(2 * hidden_size, hidden_size)
        self.nn_tmp = nn.Linear(hidden_size, hidden_size)
        self.nn_output = nn.Linear(hidden_size, 1)

    def forward(self, query_x, x):

        batch_size = x.size(0)

        #pair-wise attention
        '''transformer block'''
        query = self.query1(query_x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        x = self.nn_tmp(x)
        x = torch.squeeze(self.nn_output(x))
        return x

        #self attention
        '''transformer block'''
        query = self.query2(x).view(batch_size, -1, self.hidden_size)
        key = self.key2(x).view(batch_size, -1, self.hidden_size)
        value = self.value2(x).view(batch_size, -1, self.hidden_size)
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))
        x = torch.bmm(attention_scores, value)

        # x1 = x[:,0,:]
        # x2 = x[:,1,:]
        # x = torch.cat([x1, x2], dim=1)
        # x = self.nn_cat(x)



        return x

class GCN_Low(nn.Module):

    def __init__(self, features_size, embedding_size, low_k, bias=False):  # 16, 16

        super(GCN_Low, self).__init__()
        self.low_k = low_k
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))   #
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj_self):

        # output = torch.spmm(adj, feature)
        # output = 0.5 * output + 0.5 * feature
        conv = 0.5 * adj_self
        output = torch.spmm(conv, feature)
        for i in range(self.low_k - 1):
            output = torch.spmm(conv, output)
        ''''''

        output = torch.mm(output, self.weight)

        if self.bias is not None:
            output += self.bias
        return output

class GCN_Mid(nn.Module):

    def __init__(self, features_size, embedding_size, mid_k, bias=False):

        super(GCN_Mid, self).__init__()
        self.mid_k = mid_k
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj_self, adj_dele):

        # output = torch.spmm(adj, feature)
        # output = torch.spmm(adj, output)
        # output = 0.5 * output - 0.5 * feature
        conv = -torch.spmm(adj_self, adj_dele)
        output = torch.spmm(conv, feature)
        for i in range(self.mid_k - 1):
            output = torch.spmm(conv, output)
        ''''''

        output = torch.mm(output, self.weight)
        if self.bias is not None:
            output += self.bias

        return output


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()

        self.h_layer = n_hyper_layer

    def forward(self, embeds, i_hyper, u_hyper=None):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            # u_ret = torch.mm(u_hyper, lat)
        # return u_ret, i_ret
        return i_ret

class Hyper_Graph_Network(nn.Module):
    def __init__(self, embedding_size, hylab, n_hyper_layer, bias=False):

        super(Hyper_Graph_Network, self).__init__()
        self.hylab = hylab
        self.hyper_edge = Parameter(torch.FloatTensor(embedding_size, hylab))
        self.hgnnLayer = HGNNLayer(n_hyper_layer)
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.drop = nn.Dropout(0.5)

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.hyper_edge.size(1))
        self.hyper_edge.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, embedding, adj=None):

        item_similar = torch.mm(embedding, self.hyper_edge)
        # user_similar = torch.spmm(adj, item_similar)
        item_label = nn.functional.gumbel_softmax(item_similar, 0.2, dim=1, hard=False)
        # user_label = nn.functional.gumbel_softmax(user_similar, 0.2, dim=1, hard=False)

        item_emb = self.hgnnLayer(embedding, self.drop(item_label))
        return item_emb

class Item_Graph_Convolution(nn.Module):

    def __init__(self, features_size, embedding_size, mode, low_k, mid_k):    # 16, 16, concat
        super(Item_Graph_Convolution, self).__init__()
        self.mode = mode
        self.gcn_low = GCN_Low(features_size, embedding_size, low_k)
        self.gcn_mid = GCN_Mid(features_size, embedding_size, mid_k)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        if mode == "concat":
            self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        else:
            self.nn_cat = None

    def forward(self, feature, adj, adj_self, adj_dele):

        output_low = self.bn1(self.gcn_low(feature, adj_self))
        output_mid = self.bn2(self.gcn_mid(feature, adj_self, adj_dele))

        if self.mode == "att":
            output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        elif self.mode == "concat":
            output = (self.nn_cat(torch.cat([output_low, output_mid], dim=1)))
        elif self.mode == "mid":
            output = output_mid
        else:
            output = output_low

        return output

class SR_Rec(nn.Module):

    def __init__(self, embedding_size, price_n_bins, mode, low_k, mid_k, alpha, beta, dataset, category_emb_size=768): # 16, 20, att, 768
        super(SR_Rec, self).__init__()
        self.category_emb_size = category_emb_size  # 768
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.dataset = dataset

        self.embedding_cid2 = nn.Linear(category_emb_size, embedding_size, bias=True)    # 768 * 16
        self.embedding_cid3 = nn.Linear(category_emb_size, embedding_size, bias=True)    # 768 * 16
        self.embedding_price = nn.Embedding(price_n_bins, embedding_size)
        self.nn_emb = nn.Linear(embedding_size * 3, embedding_size)                      # (16x3) * 16
        self.item_gc = Item_Graph_Convolution(embedding_size, embedding_size, self.mode, low_k, mid_k) # 16 * 16, att
        # self.hyper_gnn = Hyper_Graph_Network(embedding_size, hylab, n_hyper_layer) # 16 * 16, att
        self.two_att = Twostage_Attention(embedding_size)                                # 16
        self.rela = nn.Bilinear(embedding_size, embedding_size, 1)

    def forward(self, features, price, adj, adj_self, adj_dele, train_set, mode='train'):

        # obtain item embeddings
        cid2 = features[:,:self.category_emb_size]
        cid3 = features[:,self.category_emb_size:]
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))

        item_latent = self.item_gc(item_latent, adj, adj_self, adj_dele)
        #
        # low_item = item_latent[:, 0, :]
        # mid_item = item_latent[:, 1, :]
        # low_emb = self.hyper_gnn(low_item)  # TODO

        # mid_emb = self.hyper_gnn(mid_item)  # TODO

        # hyper_emb = torch.stack([low_emb, mid_item], dim=1)

        # item_latent = item_latent + hyper_emb


        key_emb = item_latent[train_set[:, 0]]       # (505, 2, 16)
        pos_emb = item_latent[train_set[:, 1]]       # (505, 2, 16)
        neg_emb = item_latent[train_set[:, 2:]]      # (505, 2, 2, 16)

        low_key_emb, mid_key_emb, low_pos_emb, mid_pos_emb = key_emb[:, 0, :], key_emb[:, 1, :], pos_emb[:, 0, :], pos_emb[:, 1, :]

        low_pos = torch.sum(low_key_emb * low_pos_emb, dim=1, keepdim=True) # 6366 * 1
        mid_pos = self.rela(mid_key_emb, mid_pos_emb)    # 6366 * 1
        score_pos = torch.cat((low_pos, mid_pos), dim=1) # 6366 * 2

        for i in range(neg_emb.shape[1]):
            low_neg_emb, mid_neg_emb = neg_emb[:, i, 0, :], neg_emb[:, i, 1, :]
            if i == 0:
                low_neg = torch.sum(low_key_emb * low_neg_emb, dim=1, keepdim=True) # 6366 * 1
                mid_neg = self.rela(mid_key_emb, mid_neg_emb)    # 6366 * 1
            else:
                low_neg = torch.cat((low_neg, torch.sum(low_key_emb * low_neg_emb, dim=1, keepdim=True)), dim=1) # 6366 * 2
                mid_neg = torch.cat((mid_neg, self.rela(mid_key_emb, mid_neg_emb)), dim=1)  # 6366 * 2
        score_neg = torch.stack((low_neg, mid_neg), dim=2)


        if self.mode == "att":
            if self.dataset == 'Toys_and_Games':
                weight_pos = self.two_att(key_emb, pos_emb)  # (505, 16)
            else:
                key_latent_pos = self.two_att(pos_emb, key_emb) # (505, 16)
                pos_latent = self.two_att(key_emb, pos_emb)     # (505, 16)
                weight_pos = (key_latent_pos + pos_latent) / 2
            weight_pos = nn.functional.normalize(weight_pos, p=2, dim=1)
            weighted_score_pos = torch.sum(weight_pos * score_pos, dim=1)

            for i in range(neg_emb.shape[1]):
                neg_emb_tmp = neg_emb[:, i, :, :]
                if self.dataset != 'Toys_and_Games':
                    key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    if self.dataset != 'Toys_and_Games':
                        key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)

                else:
                    if self.dataset != 'Toys_and_Games':
                        key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)   # (505, 100, 16)

            weight_neg = (key_latent_neg + neg_latent) / 2 if self.dataset != 'Toys_and_Games' else neg_latent
            weight_neg = nn.functional.normalize(weight_neg, p=2, dim=2)
            weighted_score_neg = torch.sum(weight_neg * score_neg, dim=2)

            # pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)   # (505,)
            # neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)   # (505,2)


        if mode == 'train':
            loss_low = -torch.mean(torch.log(torch.sigmoid(low_pos - low_neg) + 1e-9))
            loss_mid = -torch.mean(torch.log(torch.sigmoid(mid_pos - mid_neg) + 1e-9))

            loss = -torch.mean(torch.log(torch.sigmoid(weighted_score_pos.unsqueeze(1) - weighted_score_neg) + 1e-9))
            return loss + self.alpha * loss_low + self.beta * loss_mid

        hr5, hr10, ndcg = self.metrics(torch.unsqueeze(weighted_score_pos, 1), weighted_score_neg)
        return hr5, hr10, ndcg

    def inference(self, features, price, adj, test_set):

        cid2 = features[:,:self.category_emb_size]
        cid3 = features[:,self.category_emb_size:]
        # concatenate the three types of embedding
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        item_latent = self.item_gc(item_latent, adj)

        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        if self.mode == "att":
            key_latent_pos = self.two_att(pos_emb, key_emb)
            pos_latent = self.two_att(key_emb, pos_emb)
            for i in range(neg_emb.shape[1]):
                neg_emb_tmp = neg_emb[:, i, :, :]
                key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                else:
                    key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)

            pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
            neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)

        else:

            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

        hr5, hr10, ndcg = self.metrics(torch.unsqueeze(pos_scores, 1), neg_scores)

        return hr5, hr10, ndcg

    def metrics(self, pos_scores, neg_scores):

        # concatenate the scores of both positive and negative samples
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)
        scores = torch.squeeze(scores)
        labels = torch.squeeze(labels)
        ranking = torch.argsort(scores, dim=1, descending=True)

        #obtain ndcg scores
        ndcg = ndcg_score(labels.cpu(), scores.cpu())

        #obtain hr scores
        k_list = [5, 10]
        hr_list = []
        for k in k_list:
            ranking_k = ranking[:, :k]
            hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))
            hr_list.append(hr)

        return hr_list[0], hr_list[1], ndcg