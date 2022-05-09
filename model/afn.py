import torch
import math
import torch.nn.functional as F
from model.layers import MultiLayerPerceptron


class LNN(torch.nn.Module):
    """
    A pytorch implementation of LNN layer
    Input shape
        - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
    Output shape
        - 2D tensor with shape:``(batch_size,LNN_dim*embedding_size)``.
    Arguments
        - **in_features** : Embedding of feature.
        - **num_fields**: int.The field size of feature.
        - **LNN_dim**: int.The number of Logarithmic neuron.
        - **bias**: bool.Whether or not use bias in LNN.
    """
    def __init__(self, num_fields, embed_dim, LNN_dim, bias=False):
        super(LNN, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.lnn_output_dim = LNN_dim * embed_dim
        self.weight = torch.nn.Parameter(torch.Tensor(LNN_dim, num_fields))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(LNN_dim, embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields, embedding_size)``
        """
        embed_x_abs = torch.abs(x) # Computes the element-wise absolute value of the given input tensor.
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        # Logarithmic Transformation
        embed_x_log = torch.log1p(embed_x_afn) # torch.log1p and torch.expm1
        lnn_out = torch.matmul(self.weight, embed_x_log)
        if self.bias is not None:
            lnn_out += self.bias
        lnn_exp = torch.expm1(lnn_out)
        output = F.relu(lnn_exp).contiguous().view(-1, self.lnn_output_dim)
        return output


class AdaptiveFactorizationNetwork(torch.nn.Module):
    def __init__(self, description, embed_dim, LNN_dim, mlp_dims, dropout, item_id_name='item_id'):
        super().__init__()
        # assert len(description) == 12, 'unillegal format of {}'.format(description)
        self.features = [name for name, _, type in description if type != 'label']
        assert item_id_name in self.features, 'unkown item id name'
        self.description = {name: (size, type) for name, size, type in description}
        self.item_id_name = item_id_name
        self.build(embed_dim, LNN_dim, mlp_dims, dropout)
    
    def build(self, embed_dim, LNN_dim, mlp_dims, dropout):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        self.embed_output_dim = 0
        self.num_fields = 0
        for name, (size, type) in self.description.items():
            if type == 'spr':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
                self.num_fields += 1
            elif type == 'ctn':
                self.ctn_linear_layer[name] = torch.nn.Linear(1, 1, bias=False)
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
                self.num_fields += 1
            elif type == 'label':
                pass
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)
        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropout)
        return

    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def only_optimize_itemid(self):
        for name, param in self.named_parameters():
            if self.item_id_name not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        return
    
    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def forward(self, x_dict):
        item_id_emb = self.emb_layer[self.item_id_name](x_dict[self.item_id_name])
        loss = self.forward_with_item_id_emb(item_id_emb, x_dict)
        return loss

    def forward_with_item_id_emb(self, item_id_emb, x_dict):
        if item_id_emb.dim() == 2:
            item_id_emb = item_id_emb.unsqueeze(1)
        linears = []
        embs = []
        for name, (_, type) in self.description.items():
            if name == self.item_id_name or type == 'label':
                continue
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x))
            elif type == 'ctn':
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                embs.append(self.emb_layer[name](x).sum(dim=1, keepdims=True))
            else:
                raise ValueError('unkwon feature: {}'.format(name))
        emb = torch.concat([item_id_emb] + embs, dim=1)
        lnn_out = self.LNN(emb)
        linear_part = torch.concat(linears, dim=1).sum(dim=1, keepdims=True)
        res = (linear_part + self.mlp(lnn_out)).squeeze(dim=1)
        return torch.sigmoid(res)

    # def forward(self, x):
    #     """
    #     :param x: Long tensor of size ``(batch_size, num_fields)``
    #     """
    #     embed_x = self.embedding(x)
    #     lnn_out = self.LNN(embed_x)
    #     x = self.linear(x) + self.mlp(lnn_out)
    #     return torch.sigmoid(x.squeeze(1))



