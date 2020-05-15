import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB
from torch_scatter import scatter_add

CUDA = torch.cuda.is_available()  # checking cuda availability


class MergeLayer(nn.Module):
    def __init__(self, h_size, device='cpu'):
        super(MergeLayer, self).__init__()
        self.weight_inbound = nn.Linear(h_size, h_size, bias=True)
        self.weight_outbound = nn.Linear(h_size, h_size, bias=True)
        self.lambda_layer = nn.Linear(h_size * 2, 1, bias=True)
        self.init_params()
        self.to(device)

    def forward(self, h_inbound, h_outbound):
        h_inbound = self.weight_inbound(h_inbound)
        h_outbound = self.weight_outbound(h_outbound)
        lambda_param = self.lambda_layer(torch.cat([h_inbound, h_outbound], dim=1))
        lambda_param = torch.sigmoid(lambda_param)
        h = lambda_param * h_inbound + (1 - lambda_param) * h_outbound
        h = F.elu(h)
        h = F.normalize(h, dim=1, p=2)
        return h

    def init_params(self):
        nn.init.xavier_normal_(self.weight_inbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.weight_outbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.lambda_layer.weight, gain=1.414)

        nn.init.zeros_(self.weight_inbound.bias)
        nn.init.zeros_(self.weight_outbound.bias)
        nn.init.zeros_(self.lambda_layer.bias)


class RelationLayer(nn.Module):
    def __init__(self, in_size, out_size, device):
        super(RelationLayer, self).__init__()
        # relation layer
        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(in_size, out_size)))
        self.weights_rel = nn.Linear(in_size, out_size, bias=False)
        self.init_params()

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, g_initial, c_ijk, edge_type):
        g = scatter_add(c_ijk, edge_type, dim=0).squeeze()
        g_prime = self.weights_rel(g_initial) + g.mm(self.W)
        g_prime = F.normalize(g_prime, p=2, dim=-1)
        return g_prime


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        if CUDA:
            dev = 'cuda'
        else:
            dev = 'cpu'
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.merge_input = MergeLayer(nhid * nheads)
        self.merge_output = MergeLayer(nhid * nheads)
        self.rel_layer = RelationLayer(relation_dim, nhid * nheads, dev)

        self.attentions_inbound = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                         nhid,
                                                         relation_dim,
                                                         dropout=dropout,
                                                         alpha=alpha,
                                                         concat=True)
                                   for _ in range(nheads)]

        self.attentions_outbound = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                          nhid,
                                                          relation_dim,
                                                          dropout=dropout,
                                                          alpha=alpha,
                                                          concat=True)
                                    for _ in range(nheads)]

        for i, attention in enumerate(self.attentions_inbound):
            self.add_module('attention_inbound_{}'.format(i), attention)

        for i, attention in enumerate(self.attentions_inbound):
            self.add_module('attention_outbound_{}'.format(i), attention)

        self.out_att_inbound = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                                     nheads * nhid, nheads * nhid,
                                                     dropout=dropout,
                                                     alpha=alpha,
                                                     concat=False
                                                     )

        self.out_att_outbound = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                                      nheads * nhid, nheads * nhid,
                                                      dropout=dropout,
                                                      alpha=alpha,
                                                      concat=False
                                                      )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        edge_list_outbound = torch.stack([edge_list[1, :], edge_list[0, :]])
        if edge_list_nhop is not None:
            edge_list_nhop_outbound = torch.stack([edge_list_nhop[1, :], edge_list_nhop[0, :]])
        else:
            edge_list_nhop_outbound = None
        x = entity_embeddings

        if edge_type_nhop is not None:
            edge_embed_nhop = relation_embed[
                                  edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        else:
            edge_embed_nhop = None
        x_in = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                          for att in self.attentions_inbound], dim=1)

        x_out = torch.cat([att(x, edge_list_outbound, edge_embed, edge_list_nhop_outbound, edge_embed_nhop)
                           for att in self.attentions_inbound], dim=1)

        x = self.merge_input(x_in, x_out)
        x = self.dropout_layer(x)

        out_relation_1 = self.rel_layer(relation_embed, edge_embed, edge_type)

        edge_embed = out_relation_1[edge_type]
        if edge_type_nhop is not None:
            edge_embed_nhop = out_relation_1[
                                  edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]
        else:
            edge_embed_nhop = None
        x_in = F.elu(self.out_att_inbound(x, edge_list, edge_embed,
                                          edge_list_nhop, edge_embed_nhop))
        x_out = F.elu(self.out_att_inbound(x, edge_list_outbound, edge_embed,
                                           edge_list_nhop_outbound, edge_embed_nhop))

        x = self.merge_output(x_in, x_out)
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha  # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]
        if not train_indices_nhop is None:
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        else:
            edge_list_nhop = None
            edge_type_nhop = None

        if (CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            if edge_list_nhop is not None:
                edge_list_nhop = edge_list_nhop.cuda()
                edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
                       mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha  # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
