from enum import Enum
import torch

class PruningTypes(Enum):
    DEPTH = 0
    WIDTH = 1
    HEAD = 2 # valid only for attention
    NONE = 3 # means doesn't do pruning

class PruningInterface:
    # make sure to implement all functions and variables in this interface
    # the way how self.nn is used is up to you, just make sure to include it
    # it could be vit from transformers or timm, you decide
    # correctly assign pruning types for your method.
    # PruningTypes.None means pruning this structure is unsupported
    def __init__(self, model, dataloader, att_prune_type: PruningTypes, mlp_prune_type: PruningTypes):
        self.model = model
        self.dataloader = dataloader
        self.att_prune_type = att_prune_type
        self.mlp_prune_type = mlp_prune_type
        self.att_importance = None
        self.mlp_importance = None

    # Your algorithm should return importance metric according to this format
    # format is designed in a way that is most efficient for that pruning type
    # code below is just a description for format of importance metrics
    # you can change it as you like. 
    # Lower importance means can be pruned earlier
    def fit(self):
        # showcase of how to store importance scores for each pruning type
        match self.att_prune_type:
            case PruningTypes.DEPTH:
                self.att_importance = torch.randn((self.nn.n_blocks,))
            case PruningTypes.HEAD :
                self.att_importance = [torch.randn((b.n_heads,)) for b in self.nn.blocks]
            case PruningTypes.WIDTH:
                # [q, k] and [v, proj] matrices are interrelated
                # their neurons should be pruned together
                # so we only need 2 tensors per block instead of 4
                self.att_importance = [[
                    # q, v shape is (batch, n_seq, n_heads, n_dim)
                    torch.randn(b.q.shape[2:]), torch.randn(b.v.shape[2:])
                ] for b in self.nn.blocks]
                # width pruning in this case should reduce hidden dimension of 
                # q,k,v matrices. Not embedding dimension.
                # reducing embedding dimension is possible, but troublesome
            case _:
                self.att_importance = None

        match self.mlp_prune_type:
            case PruningTypes.DEPTH:
                self.mlp_importance = torch.randn((self.nn.n_blocks,))
            case PruningTypes.WIDTH:
                # fc1 and fc2 are interrelated, their neurons are pruned together
                # so we only need 1 tensor per block instead of 2
                self.mlp_importance = [torch.randn(b.fc1.shape[:1]) for b in self.nn.blocks]
                # same consideration goes for reducing embedding dimension
                # just like in attention width pruning
            case _:
                self.mlp_importance = None
    
    def get_mlp_importance(self):
        if self.mlp_prune_type != PruningTypes.NONE and self.mlp_importance is None: 
            self.fit()
        return self.mlp_neuron_importance

    def get_att_importance(self):
        if self.att_prune_type != PruningTypes.NONE and self.att_importance is None: 
            self.fit()
        return self.att_importance
    


class ExamplePruningMethod(PruningInterface):
    def __init__(self, model, dataloader):
        # initializes required parameters
        super.__init__(self, model, dataloader, PruningTypes.DEPTH, PruningTypes.WIDTH)
        # you can add whatever else you need

    def fit(self):
        # implement a fit function yourself to compute importance scores
        self.att_importance = torch.randn((self.nn.n_blocks,))
        self.mlp_importance = [torch.randn(b.fc1.shape[:1]) for b in self.nn.blocks]