#from .embedding import EMBEDDINGS
#from .embedding import embed_nohub as EMBEDDINGS
from .pca import get_pca_weights


global args




def get_extra_tensors(feature):
    '''
    if args.use_cached:
        # No need to create a train loader when we're using cached features
        train_loader = None
        train_dataset = None
    else:
        train_dataset = get_dataset(name=args.dataset, split="train")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, drop_last=False
        )


    requires_pca_weights = \
        ((args.embedding == "nohub") and (args.nohub.pca_mode == "base")) \
        or ((args.embedding == "nohubs") and (args.nohubs.pca_mode == "base")) \
        or ((args.embedding == "pca") and (args.pca.mode == "base")) \
        or ((args.embedding == "l2") and (args.l2.pca_mode == "base")) \
        or ((args.embedding == "cl2") and (args.cl2.pca_mode == "base"))

    requires_mean = (args.embedding == "cl2") and (args.cl2.center_mode == "base")

    requires_features = (args.embedding == "tcpr")
    '''
    #他的model是什么 是resnet 这里是它提取的feature吗？？他的feature好像有点复杂

    train_features = feature

    #"train_mean": train_features.mean(dim=args.cl2.dim, keepdims=True)[None] if requires_mean else None,
    # if requires_pca_weights else None,
    extra_tensors = {
        "train_mean":  None,
        "train_pca_weights": get_pca_weights(train_features)[None],
        "train_features": train_features
    }
    return extra_tensors