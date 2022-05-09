import torch

from model.layers import FactorizationMachine, MultiLayerPerceptron


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)

class ProductNeuralNetworkModel(torch.nn.Module):

    def __init__(self, description, embed_dim, mlp_dims, dropout, method='inner', item_id_name='item_id'):
        super().__init__()
        # assert len(description) == 12, 'unillegal format of {}'.format(description)
        self.features = [name for name, _, type in description if type != 'label']
        assert item_id_name in self.features, 'unkown item id name'
        self.description = {name: (size, type) for name, size, type in description}
        self.item_id_name = item_id_name
        self.build(embed_dim, mlp_dims, dropout, method)
    
    def build(self, embed_dim, mlp_dims, dropout, method):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        self.embed_output_dim, self.num_fields = 0, 0
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
        self.mlp = MultiLayerPerceptron(self.num_fields * (self.num_fields - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)
        if method == 'inner':
            self.pn = InnerProductNetwork()
        elif method == 'outer':
            self.pn = OuterProductNetwork(self.num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)
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
        cross_term = self.pn(emb)

        x = torch.concat([emb.view(-1, self.embed_output_dim), cross_term], dim=1)
        res = self.mlp(x).squeeze(dim=1)
        return torch.sigmoid(res)

