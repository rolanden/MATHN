#import src.config
#from src.config.parser.types import *
from functools import partial
from pathlib import Path
#from embeddings import EMBEDDING_ARG_ADDERS
#from classifiers import CLASSIFIER_ARG_ADDERS
#import config


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


def add_base_args(parser):
    parser.add_argument_with_default("--dataset", choices=["debug", "mini", "tiered"])
    parser.add_argument_with_default("--embedding", type=str_lower, choices=["nohub"])
    #parser.add_argument_with_default("--classifier", type=str_lower, choices=CLASSIFIER_ARG_ADDERS.keys())

    # Encoder model args
    parser.add_argument_with_default("--arch", type=str_lower, choices=["debug", "resnet18", "wrn_s2m2"])
    parser.add_argument_with_default("--checkpoint", type=path_or_none)

    # Feature extraction args
    parser.add_argument_with_default("--batch_size", default=128, type=int)
    parser.add_argument_with_default("--n_workers", default=4, type=int)
    parser.add_argument_with_default("--use_cached", default=False, type=str_to_bool)
    parser.add_argument_with_default("--cache_dir", default=None, type=path_or_none)

    # Evaluation args
    parser.add_argument_with_default("--n_episodes", default=10000, type=int)
    parser.add_argument_with_default("--episode_batch_size", default=500, type=int_or_none)
    parser.add_argument_with_default("--n_shots", default=1, type=int)
    parser.add_argument_with_default("--n_ways", default=5, type=int)
    parser.add_argument_with_default("--n_queries", default=15, type=int)
    parser.add_argument_with_default("--eval_split", default="test", choices=["val", "test"])

    # Misc args
    parser.add_argument_with_default("--log_level", default="INFO", type=str_upper,
                                     choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"])
    parser.add_argument_with_default("--wandb_tags", default=None, type=str_or_none)


def add_hubness_args(parser):
    # Enable/Disable hubness computations
    parser.add_argument_with_default("--compute_hubness", default=True, type=str_to_bool)
    # K-occurrence args
    parser.add_argument_with_default("--k_occ.k", type=int)
    parser.add_argument_with_default("--k_occ.metric", type=str)


def parse_args():
    parser = config.parser.ArgumentParserWithDefaults(parse_config_file=True)

    # Parse the base args, so we can add new sub-args based on the values.
    add_base_args(parser)
    base_args, _ = parser.parse_known_args()

    # Embedding args
    EMBEDDING_ARG_ADDERS[base_args.embedding](parser)
    # Classifier args
    CLASSIFIER_ARG_ADDERS[base_args.classifier](parser)
    # Hubness args
    add_hubness_args(parser)

    # Parse all args
    args = parser.parse_args(check=True)
    return args
