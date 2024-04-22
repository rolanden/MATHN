#from config.parser.types import *
from no_hubness.no_hub import  NoHub
from no_hubness.no_hub_base import train_no_hub
from functools import partial
from pathlib import Path
from no_hubness.embeding_feature import get_extra_tensors
import torch.nn as nn

global args
import torch

def _type_or_none(arg, typ=str):
    if (arg is None) or (arg.lower() == "none"):
        return None
    return typ(arg)


str_or_none = partial(_type_or_none, typ=str)
int_or_none = partial(_type_or_none, typ=int)
float_or_none = partial(_type_or_none, typ=float)
path_or_none = partial(_type_or_none, typ=Path)


def str_upper(arg):
    if arg is None:
        return None
    return arg.upper()


def str_lower(arg):
    if arg is None:
        return None
    return arg.lower()


def str_to_bool(arg):
    arg = str_lower(arg)
    if arg in {"true", "t", "yes", "y"}:
        return True
    return False

def _add_base_no_hub_args(parser, prefix):
    parser.add_argument_with_default(f"--{prefix}.init", type=str_lower)
    parser.add_argument_with_default(f"--{prefix}.out_dims", type=int_or_none)
    parser.add_argument_with_default(f"--{prefix}.initial_dims", type=int_or_none)
    parser.add_argument_with_default(f"--{prefix}.kappa", type=float)
    parser.add_argument_with_default(f"--{prefix}.perplexity", type=float)
    parser.add_argument_with_default(f"--{prefix}.re_norm", type=str_to_bool)
    parser.add_argument_with_default(f"--{prefix}.eps", type=float)
    parser.add_argument_with_default(f"--{prefix}.p_sim", type=str_lower)
    parser.add_argument_with_default(f"--{prefix}.p_rel_tol", type=float_or_none)
    parser.add_argument_with_default(f"--{prefix}.p_abs_tol", type=float_or_none)
    parser.add_argument_with_default(f"--{prefix}.p_betas", type=float_or_none, nargs=2)
    parser.add_argument_with_default(f"--{prefix}.pca_mode", type=str, choices=["episode", "base"])
    parser.add_argument_with_default(f"--{prefix}.n_iter", type=int)
    parser.add_argument_with_default(f"--{prefix}.learning_rate", type=float)


def add_no_hub_args(parser):
    _add_base_no_hub_args(parser, prefix="nohub")
    parser.add_argument_with_default("--nohub.loss_weights", type=float, nargs="+")


EMBEDDING_ARG_ADDERS = {
    "nohub": add_no_hub_args,

}

def embed_nohub(features,  episode):
    features = features.float
    extra_tensors = get_extra_tensors(features).detach()
    no_hub = NoHub(features, pca_weights=extra_tensors["train_pca_weights"])
    #把这里面的训练过程搬出来
    embeddings, losses = train_no_hub(no_hub, global_step=episode)
    return embeddings,losses

EMBEDDINGS = {
    "nohub": embed_nohub,
}

