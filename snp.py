from tqdm import tqdm
from copy import deepcopy
from abc import ABC
import torch, timm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2

_ = torch.set_grad_enabled(False)


# common variables: device
device = torch.device("cpu")
#     "mps" if torch.backends.mps.is_available() else (
#     "cuda" if torch.cuda.is_available() else
#     "cpu"
# ))
seed = 42


# utility functions
@torch.no_grad()
def test_model(model, dataloader):
    sm = torch.nn.Softmax(dim=1)
    acc, correct = 0, 0
    for features, labels in tqdm(iter(dataloader)):
        features = features.to(device)
        labels = labels.to(device)
        clf = sm(model(features)).argmax(1)
        correct += (clf == labels).sum()
    acc = correct / len(dataloader.dataset)
    return acc

# importance scoring functions
def svd_score(q, k):
    """q, k shape (batch, head, sequence, dimension)"""
    _, RQ = torch.linalg.qr(q.cpu(), mode='r')
    _, RK = torch.linalg.qr(k.cpu(), mode='r')
    Ug, S, Vhg = torch.linalg.svd(RQ @ RK.mT, full_matrices=False)
    RQ, RK = RQ.to(ref := q), RK.to(ref)
    Uhg, Vhg = Ug.mT.to(ref), Vhg.to(ref)    

    F.normalize(RQ, dim=2, out=RQ)
    F.normalize(RK, dim=2, out=RK)
    A, B = Uhg @ RQ, Vhg @ RK
    torch.multiply(A, B, out=A)
    torch.abs(A, out=A)
    return A.sum(-2)

def similarity_score(x):
    """input shape (B, N, D) - batch, sequence len, embedding dimension"""
    x = F.normalize(x, dim=1)
    score = torch.einsum('bnd,bnt->bdt', x, x)
    torch.abs(score, out=score)
    score = x.shape[-1] - score.sum(dim=-1) 
    return score


# attention block with split qk & v matrices, convert_attention_blocks
class Attention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.fused_attn = attn.fused_attn
        
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        # qkv.weight.shape: (num_heads * (2*qk_dim + v_dim), emb_dim)
        self.qkv = attn.qkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
        
        # previous attention class assumed 
        # that all head dims for Q, K, V are the same
        # between heads and between Q, K, V
        # this attention class doesn't assume
        # that head dims are the same between Q, K, V
        # but, still, they're the same between heads of Q, K, V
        self.qk_dim, self.v_dim = attn.head_dim, attn.head_dim
        self.split_dim = [self.num_heads*c for c in [self.qk_dim]*2+[self.v_dim]]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = torch.split(self.qkv(x), self.split_dim, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.qk_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.qk_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.v_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.split_dim[-1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def convert_attention_blocks(model):
    for i, b in enumerate(model.blocks):
        model.blocks[i].attn = Attention(b.attn)
    return model


# hooks and hooks management
class SNP:
    def __init__(self, attn, do_svd_scoring=True, do_sim_scoring=True):
        self.num_heads = attn.num_heads
        self.qk_dim = attn.qk_dim
        self.v_dim = attn.v_dim
        self.split_dim = attn.split_dim
        self.scale = attn.scale
        self.do_svd_scoring = do_svd_scoring
        self.do_sim_scoring = do_sim_scoring
        self.svd_scores = 0
        self.sim_scores = 0

    def __call__(self, module, inp, out):
        B, N = out.shape[:2]
        q, k, v = torch.split(out, self.split_dim, dim=-1)
        q = q.view(B, N, self.num_heads, self.qk_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.qk_dim).transpose(1, 2)
        if self.do_svd_scoring:
            self.svd_scores = svd_score(q, k)
        if self.do_sim_scoring:
            self.sim_scores = similarity_score(v).view(B, self.num_heads, self.v_dim)

class MLP_SNP:
    def __init__(self, module, do_scoring=True):
        self.out_features = module.out_features
        self.do_scoring = do_scoring
        self.sim_scores = 0
        
    def __call__(self, module, inp, out):
        if self.do_scoring:
            self.sim_scores = similarity_score(out)

def register_hooks(model, scoring_flags):
    """
        scoring_flags: (SVD [QK], cos-sim [V], cos-sim [MLP])
    """
    for b in model.blocks:
        snp = SNP(b.attn, *scoring_flags[:2])
        b.attn.qkv.register_forward_hook(snp)
        mlp = MLP_SNP(b.mlp.fc1, scoring_flags[2])
        b.mlp.fc1.register_forward_hook(mlp)
    return model

def remove_hooks(model):
    for b in model.blocks:
        b.attn.qkv._forward_hooks.popitem()
        b.mlp.fc1._forward_hooks.popitem()
    return model


# collect_scores, zero_out (weights)
def collect_scores(model, dl):
    svd_scores, sim_scores, mlp_scores = [], [], []
    snp_hooks, mlp_hooks = [], []
    for b in model.blocks:
        snp_hooks.append(next(iter(b.attn.qkv._forward_hooks.values())))
        mlp_hooks.append(next(iter(b.mlp.fc1._forward_hooks.values())))
        hook = snp_hooks[-1]
        svd_scores.append(torch.zeros((hook.num_heads, hook.qk_dim), device=device))
        sim_scores.append(torch.zeros((hook.num_heads, hook.v_dim), device=device))
        mlp_scores.append(torch.zeros((mlp_hooks[-1].out_features,), device=device))
    
    with torch.no_grad():
        for features, _ in tqdm(dl):
            model(features.to(device))
            for i, (h1, h2) in enumerate(zip(snp_hooks, mlp_hooks)):
                if h1.do_svd_scoring: svd_scores[i] += h1.svd_scores.sum(0)
                if h1.do_sim_scoring: sim_scores[i] += h1.sim_scores.sum(0)
                if h2.do_scoring: mlp_scores[i] += h2.sim_scores.sum(0)

    return svd_scores, sim_scores, mlp_scores


class SNPPruning:
    """
    UNIFIED INTERFACE for plugging in ANY pruning method

    FLEXIBLE ARCHITECTURE:
    - You can provide ONLY mlp_neuron_importance (MLP-only pruning)
    - You can provide ONLY att_importance (Attention-only pruning)
    - You can provide BOTH (prune both components)
    """

    def __init__(self, model, attention_granularity=None, dataloader=None):
        self.model = convert_attention_blocks(deepcopy(model))
        self.attention_granularity = 'width'
        self.dataloader = dataloader

        if attention_granularity is not None:
            attention_granularity = attention_granularity.lower()
            if attention_granularity not in ['width', 'depth', 'head']:
                raise ValueError(f"Invalid attention_granularity: {attention_granularity}")

        self.mlp_neuron_importance = None
        self.att_importance = None

    def compute_importance_scores(self):
        register_hooks(self.model, [True, True, True])
        iscores = collect_scores(self.model, self.dataloader)
        remove_hooks(self.model)
        for i, (a, b) in enumerate(zip(iscores[0], iscores[1])):
            a = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)
            b = (b - b.mean(1, keepdim=True)) / b.std(1, keepdim=True)
            iscores[0][i] = (a + b).flatten()

        self.mlp_neuron_importance = iscores[2]
        self.att_importance = iscores[0]

    def get_mlp_importance(self):
        if self.mlp_neuron_importance is None:
            self.compute_importance_scores()
        return self.mlp_neuron_importance

    def get_att_importance(self):
        if self.att_importance is None:
            self.compute_importance_scores()
        return self.att_importance
