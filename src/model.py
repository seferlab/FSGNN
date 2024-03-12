import torch
import dgl
import dgl.nn as dglnn
import copy
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import hinge_loss
import dgl.function as fn

class FEAST_layer(nn.Module):
    def __init__(self, input_dim, output_dim, head, relation_aware, etype, dropout, device, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = output_dim
        self.if_sum = if_sum
        self.device = device
        self.input_dim = input_dim
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(input_dim, output_dim*head)
        self.w_linera = nn.Linear(input_dim, output_dim*head)
        if not if_sum:
            self.w_liner_2 = nn.Linear(input_dim, output_dim*head)
            self.w_linera_2 = nn.Linear(input_dim, output_dim*head)
        else:
            self.w_liner_2 = nn.Linear(input_dim, output_dim)
            self.w_linera_2 = nn.Linear(input_dim, output_dim)
        self.atten_pos = nn.Linear(self.hd * 2, 1)
        self.atten_neg = nn.Linear(self.hd * 2, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, g, h, ah):
        with g.local_scope():
            g.ndata['sfeat'] = h
            g.ndata['afeat'] = ah
            g.apply_edges(self.sign_edges, etype=self.etype)
            th = self.w_liner(h)
            tah = self.w_linera(ah)
            lh = self.w_liner_2(h)
            lah = self.w_linera_2(ah)
            g.ndata['h'] = th
            g.ndata['ah'] = tah
            g.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = g.ndata['out']
            aout = g.ndata['aout']
            out = out + lh
            aout = aout + lah
            return out, aout

    def message(self, edges):
        src = edges.src
        dst = edges.dst
        edge_similar = edges.data['sign']
        src_pos_features = src['h'][edge_similar >= 0].view(-1, self.head, self.hd)
        src_pos_features_op = src['ah'][edge_similar >= 0].view(-1, self.head, self.hd)
        src_neg_features = src['h'][edge_similar < 0].view(-1, self.head, self.hd)
        src_neg_features_op = src['ah'][edge_similar < 0].view(-1, self.head, self.hd)
        dst_pos_features = dst['h'][edge_similar >= 0].view(-1, self.head, self.hd)
        dst_pos_features_op = dst['ah'][edge_similar >= 0].view(-1, self.head, self.hd)
        dst_neg_features = dst['h'][edge_similar < 0].view(-1, self.head, self.hd)
        dst_neg_features_op = dst['ah'][edge_similar < 0].view(-1, self.head, self.hd)
        z_pos = torch.cat([src_pos_features, dst_pos_features], dim=-1)
        z_pos_op = torch.cat([src_pos_features_op, dst_pos_features_op], dim=-1)
        z_neg = torch.cat([src_neg_features, dst_neg_features_op], dim=-1)
        z_neg_op = torch.cat([src_neg_features_op, dst_neg_features], dim=-1)
        alpha_pos = self.atten_pos(z_pos)
        alpha_pos = self.leakyrelu(alpha_pos)
        alpha_pos_op = self.atten_pos(z_pos_op)
        alpha_pos_op = self.leakyrelu(alpha_pos_op)
        alpha_neg = self.atten_neg(z_neg)
        alpha_neg = self.leakyrelu(alpha_neg)
        alpha_neg_op = self.atten_neg(z_neg_op)
        alpha_neg_op = self.leakyrelu(alpha_neg_op)

        num = src['h'].shape[0]
        sf = torch.zeros((num, self.head, self.hd)).to(self.device)
        sfo = torch.zeros((num, self.head, self.hd)).to(self.device)
        atten = torch.zeros((num, self.head, 1)).to(self.device)
        atten_op = torch.zeros((num, self.head, 1)).to(self.device)
        mask_index_pos = torch.arange(0, num)[edge_similar >= 0].to(self.device)
        mask_index_neg = torch.arange(0, num)[edge_similar < 0].to(self.device)
        sf.index_add_(0, mask_index_pos, src_pos_features)
        sf.index_add_(0, mask_index_neg, src_neg_features_op)
        sfo.index_add_(0, mask_index_pos, src_pos_features_op)
        sfo.index_add_(0, mask_index_neg, src_neg_features)
        atten.index_add_(0, mask_index_pos, alpha_pos)
        atten.index_add_(0, mask_index_neg, alpha_neg_op)
        atten_op.index_add_(0, mask_index_pos, alpha_pos_op)
        atten_op.index_add_(0, mask_index_neg, alpha_neg)
        
        return {'atten':atten, 'atten_op':atten_op, 'sf':sf, 'sfo':sfo}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        alpha_op = nodes.mailbox['atten_op']
        sf = nodes.mailbox['sf']
        sfo = nodes.mailbox['sfo']
        alpha = self.softmax(alpha)
        alpha_op = self.softmax(alpha_op)
        out_n = torch.sum(alpha*sf, dim=1)
        out_op = torch.sum(alpha_op*sfo, dim=1)
        if not self.if_sum:
            out_n = out_n.view(-1, self.head*self.hd)
            out_op = out_op.view(-1, self.head*self.hd)
        else:
            out_n = out_n.sum(dim=-2)
            out_op = out_op.sum(dim=-2)
        return {'out':out_n, 'aout': out_op}

    def sign_edges(self, edges):
        src = torch.cat([edges.src['sfeat'],edges.src['afeat']],1)
        dst = torch.cat([edges.dst['sfeat'],edges.dst['afeat']],1)
        score = self.relation_aware(src, dst)
        return {'sign':torch.sign(score)}


class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(2*input_dim, output_dim)
        self.f_liner = nn.Linear(3*output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


class MultiRelationFEASTLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, dataset, dropout, device, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove('homo')
        self.n_relation = len(self.relation)
        self.if_sum = if_sum
        if not if_sum:
            self.liner = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
            self.linera = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
        else:
            self.liner = nn.Linear(self.n_relation*output_dim*2, output_dim*2)
        self.relation_aware = RelationAware(input_dim, output_dim*head, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relation:
            self.minelayers[e] = FEAST_layer(input_dim, output_dim, head, self.relation_aware, e, dropout, device, if_sum)
        self.epsilon = torch.FloatTensor([1e-12]).to(device)
    
    def forward(self, g, h, ah):
        phs = []
        ophs = []
        for e in self.relation:
            he, ahe = self.minelayers[e](g, h, ah)
            phs.append(he)
            ophs.append(ahe)
        h = torch.cat(phs, dim=1)
        ah = torch.cat(ophs, dim=1)
        if not self.if_sum:
            h = self.dropout(h)
            h = self.liner(h)
            ah = self.dropout(ah)
            ah = self.linera(ah)
            return h, ah
        else:
            h = torch.cat([h, ah], 1)
            h = self.dropout(h)
            h = self.liner(h)
            return h
    
    def loss(self, g, h, ah):
        with g.local_scope():
            g.ndata['sfeat'] = h
            g.ndata['afeat'] = ah

            g.apply_edges(self.score_edges, etype='homo')
            edges_score = g.edges['homo'].data['score']
            edge_train_mask = g.edges['homo'].data['train_mask'].bool()
            edge_train_label = g.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            if not self.if_sum:
                agg_h, agg_ah = self.forward(g, h, ah)
                return agg_h, agg_ah, edge_diff_loss
            else:
                agg_h = self.forward(g, h, ah)
                return agg_h, edge_diff_loss

    
    def score_edges(self, edges):
        src = torch.cat([edges.src['sfeat'],edges.src['afeat']],1)
        dst = torch.cat([edges.dst['sfeat'],edges.dst['afeat']],1)
        score = self.relation_aware(src, dst)
        return {'score':score}

    
class FEAST(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        self.n_layer = args.n_layer
        self.input_dim = g.nodes['r'].data['feature'].shape[1]
        self.intra_dim = args.intra_dim
        self.n_class = args.n_class
        self.gamma = args.gamma
        self.n_layer = args.n_layer
        self.preprocess = nn.Linear(g.nodes['r'].data['feature'].shape[1], args.intra_dim)
        self.mine_layers = nn.ModuleList()
        if args.n_layer == 1:
            self.mine_layers.append(MultiRelationFEASTLayer(self.input_dim, 1, args.head, g, args.dropout, args.device, if_sum=True))
        else:
            self.mine_layers.append(MultiRelationFEASTLayer(self.input_dim, self.intra_dim, args.head, g, args.dropout, args.device))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationFEASTLayer(self.intra_dim*args.head, self.intra_dim, args.head, g, args.dropout, args.device))
            self.mine_layers.append(MultiRelationFEASTLayer(self.intra_dim*args.head, 1, args.head, g, args.dropout, args.device, if_sum=True))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.predictor = nn.Sequential(
            nn.Linear(self.intra_dim, self.intra_dim),
            nn.ReLU(),
            nn.Linear(self.intra_dim, self.intra_dim),
            nn.ReLU(),
            nn.Linear(self.intra_dim, 1))
        self.liner = nn.Linear(self.intra_dim, self.n_class)
        self.epsilon = torch.FloatTensor([1e-12]).to(args.device)

    def forward(self, g):
        sfeats = g.ndata['feature'].float()
        afeats = torch.zeros_like(sfeats)
        if self.n_layer == 1:
            h = self.mine_layers[0](g, sfeats, afeats)
        else:
            h, ah = self.mine_layers[0](g, sfeats, afeats)
            h = self.relu(h)
            ah = self.relu(ah)
            h = self.dropout(h)
            ah = self.dropout(ah)
            for i in range(1, len(self.mine_layers)-1):
                h, ah = self.mine_layers[i](g, h, ah)
                h = self.relu(h)
                ah = self.relu(ah)
                h = self.dropout(h)
                ah = self.dropout(ah)
            h = self.mine_layers[-1](g, h, ah)
        return h
    
    def loss(self, g):
        sfeats = g.ndata['feature'].float()
        afeats = torch.zeros_like(sfeats)
        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        if self.n_layer == 1:
            h, edge_loss = self.mine_layers[0].loss(g, sfeats, afeats)
        else:
            h, ah, edge_loss = self.mine_layers[0].loss(g, sfeats, afeats)
            h = self.relu(h)
            ah = self.relu(ah)
            h = self.dropout(h)
            ah = self.dropout(ah)
            for i in range(1, len(self.mine_layers)-1):
                h, ah, e_loss = self.mine_layers[i].loss(g, h, ah)
                h = self.relu(h)
                ah = self.relu(ah)
                h = self.dropout(h)
                ah = self.dropout(ah)
                edge_loss += e_loss
            h, e_loss = self.mine_layers[-1].loss(g, h, ah)
            edge_loss += e_loss

        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        
        loss = model_loss + self.gamma*edge_loss
        return loss
