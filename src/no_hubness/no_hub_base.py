import logging
from abc import ABC, abstractmethod
import torch as th
from torch import nn
import os
#import config
from . import helpers

import wandb
#from wandb_utils import wandb_logger
#from pca import get_pca_weights
#from .util import x2p

def get_experiment_id():
    try:
        eid = os.environ["EXPERIMENT_ID"]
    except KeyError:
        eid = wandb.util.generate_id()
        os.environ["EXPERIMENT_ID"] = eid
        logger.warning(f"Could not find EXPERIMENT_ID in environment variables. Using generated id '{eid}'.")
    return eid


'''

class _WandBLogger:
    def __init__(self):
        # Below values will be set when `init` is called.
        self.name = None
        self.args = None
        self.run = None

        # Constants from environment variables
        self.eid = None
        self.entity = os.environ["WANDB_ENTITY"]
        self.project = os.environ["WANDB_PROJECT"]

        self._accumulated_logs = {}

    def init(self, args, job_type="evaluate", tags=None):
        self.eid = get_experiment_id()
        self.name = f"{args.dataset}-{args.arch}-{args.embedding}-{args.classifier}-{self.eid}"

        if args.wandb_tags is not None:
            tags = (tags or []) + self._parse_tags(args.wandb_tags)

        cfg = args.to_dict()
        cfg.update(**helpers.versions())
        del cfg["wandb_tags"]

        init_kwargs = dict(
            name=self.name,
            job_type=job_type,
            config=cfg,
            entity=self.entity,
            project=self.project,
            tags=tags,
            reinit=True,
        )

        try:
            self.run = wandb.init(**init_kwargs)
        except wandb.errors.UsageError as err:
            logger.warning(f"Got error: '{str(err)}' when calling wandb.init. Attempting to init with "
                           f"'settings=wandb.Settings(start_method=''fork'')'")
            self.run = wandb.init(settings=wandb.Settings(start_method="fork"), **init_kwargs)

        return self.run

    @staticmethod
    def _parse_tags(tag_str):
        # Assumes comma-delimited tags
        tags = [tag.strip() for tag in tag_str.split(",")]
        return tags

    def accumulate(self, dct, global_step, local_step, max_local_steps):
        total_step = (global_step * max_local_steps) + local_step
        if total_step in self._accumulated_logs:
            self._accumulated_logs[total_step].update(dct)
        else:
            self._accumulated_logs[total_step] = dct

    def log_accumulated(self):
        for step, logs in sorted(self._accumulated_logs.items(), key=lambda item: item[0]):
            self.run.log(logs, step=step)


wandb_logger = _WandBLogger()
'''

def get_pca_weights(inputs):
    logger.debug(f"Running PCA on inputs with shape = {inputs.size()}.")

    if inputs.ndim == 2:
        drop_first_dim = True
        # Insert a dummy first dimension to emulate batched computation with batch size = 1
        inputs = inputs[None, :, :]
    else:
        drop_first_dim = False

    # Center data
    inputs -= inputs.mean(dim=1, keepdim=True)
    # Compute eigenvectors
    _, eig_vec = th.linalg.eigh(inputs.transpose(1, 2) @ inputs)

    # Eigenvectors are ordered by ascending eigenvalues, so we flip it to descending. This way the principal components
    # that explain the most variance are first in the array.
    eig_vec = th.flip(eig_vec, dims=[2])

    if drop_first_dim:
        eig_vec = eig_vec[0]

    return eig_vec


def x2p(x, perplexity, sim="rbf", betas=(None, None), max_iter=20, rel_tol=1e-2, abs_tol=None, eps=1e-12,
        return_dist_and_betas=False):
    n_episodes, n_samples, _ = x.shape
    if sim == "rbf":
        dist = th.cdist(x, x)
        max_dist = 1e12
    elif sim == "vmf":
        x_norm = nn.functional.normalize(x, dim=2, p=2)
        dist = - (x_norm @ x_norm.transpose(1, 2))
        max_dist = 1
    elif sim == "precomputed":
        dist = x
        max_dist = dist.max()
    else:
        raise RuntimeError(f"Unknown similarity function in x2p: '{sim}'")





device = 'cuda' if th.cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)


class NoHubBase(ABC, nn.Module):
    def __init__(self, inputs, *, init="pca", out_dims=32, initial_dims=None, kappa=2.0, perplexity=30.0, re_norm=True,
                 eps=1e-12, p_sim="vmf", p_rel_tol=1e-2, p_abs_tol=None, p_betas=(None, None), pca_mode="base",
                 pca_weights=None, loss_weights=None, learning_rate=1e-1, n_iter=50):
        super(NoHubBase, self).__init__()

        self.re_norm = re_norm
        self.eps = th.tensor(eps, divice='cuda')
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.perplexity = perplexity
        self.p_sim = p_sim
        self.p_rel_tol = p_rel_tol
        self.p_abs_tol = p_abs_tol
        self.eps = th.tensor(eps, divice='cuda')
        self.p_betas = p_betas

        # Passing out_dims=None causes embeddings to have same dimensionality as inputs
        self.out_dims = out_dims if out_dims is not None else inputs.size(2)

        assert inputs.ndim == 3, f"NoHub expected inputs tensor with shape (episodes, samples, feature). " \
                                 f"Got: '{inputs.size()}.'"
        self.embedding_size = (inputs.size(0), inputs.size(1), self.out_dims)

        # Determine the pca-weights (transformation) based on the given pca_mode
        self.set_pca_weights(pca_mode=pca_mode, pca_weights=pca_weights, inputs=inputs)
        # Initialize embeddings
        self.init_embeddings(init=init, out_dims=self.out_dims, inputs=inputs)
        # Run PCA to pre-process inputs?
        self.preprocess_inputs(initial_dims=initial_dims, inputs=inputs)
        # Compute P-values
        self.set_p()

        # Initialize losses
        self.losses = self.init_losses()
        # Loss weights
        self.set_loss_weights(loss_weights, len(self.losses))

    def set_pca_weights(self, pca_mode, pca_weights, inputs):
        if pca_mode == "base":
            assert pca_weights is not None, "pca_mode='base' requires pca_weights to be not None."
            self.pca_weights = pca_weights
        elif pca_mode == "episode":
            if pca_weights is not None:
                logger.warning(f"Argument 'pca_weights' is ignored when pca_mode='episode'.")
            self.pca_weights = get_pca_weights(inputs)

    def init_embeddings(self, init, out_dims, inputs):
        if init == "random":
            initial_embeddings = th.randn(size=self.embedding_size,device='cuda')
        elif init == "pca":
            initial_embeddings = inputs @ self.pca_weights[:, :, :out_dims]
        else:
            raise RuntimeError(f"Unknown init strategy for NoHub: '{init}'.")

        initial_embeddings = nn.functional.normalize(initial_embeddings, dim=-1, p=2)
        self.register_parameter(name="embeddings", param=nn.Parameter(initial_embeddings, requires_grad=True))

    def preprocess_inputs(self, initial_dims, inputs):
        if initial_dims is None:
            self.inputs = inputs
        else:
            self.inputs = inputs @ self.pca_weights[:, :, :initial_dims]

    def set_loss_weights(self, loss_weights, n_loss_terms):
        if loss_weights is None:
            self.loss_weights = n_loss_terms * [1 / n_loss_terms]
        else:
            assert len(loss_weights) == n_loss_terms, f"Expected loss weights to have same length as losses. " \
                                                      f"Got {len(loss_weights)} != {n_loss_terms}."
            self.loss_weights = loss_weights

    def set_p(self):
        self.p = x2p(self.inputs, perplexity=self.perplexity, sim=self.p_sim, rel_tol=self.p_rel_tol,
                     abs_tol=self.p_abs_tol, eps=self.eps, betas=self.p_betas)

    @abstractmethod
    def init_losses(self):
        # Should return a list of loss modules.
        pass

    @th.no_grad()
    def update_embeddings(self, new_embeddings):
        self.embeddings.copy_(new_embeddings)

    def forward(self):
        return self.embeddings

    def loss(self):
        return sum([weight * loss(self) for weight, loss in zip(self.loss_weights, self.losses)])

    def train_step(self, optimizer):
        optimizer.zero_grad()
        embeddings = self()
        loss = self.loss()
        #loss.backward()
        #optimizer.step()

        if self.re_norm:
            normed_embeddings = nn.functional.normalize(embeddings, dim=2, p=2)
            self.update_embeddings(normed_embeddings)

        return loss.detach()


def train_no_hub(no_hub, log_interval=10, profiler=None, global_step=0, log_wandb=True):
    opt = th.optim.Adam(params=no_hub.parameters(), lr=no_hub.learning_rate, betas=(0.9, 0.999))
    losses = th.zeros(no_hub.n_iter, device='cuda')

    for i in range(1):      #(no_hubness.n_iter):
        loss = no_hub.train_step(optimizer=opt)
        losses[i] = loss

        if i % log_interval == 0:
            # Log to WandB and console
            _loss = helpers.npy(loss)
            logger.debug(f"NoHub-iter = {i} - Loss = {_loss}")
            if log_wandb:
                wandb_logger.accumulate({"loss.NoHub": _loss}, global_step=global_step, local_step=i,
                                        max_local_steps=no_hub.n_iter)

        if profiler is not None:
            profiler.step()

    if th.any(th.isnan(losses[-1])):
        logger.warning(f"NoHub resulted in nan loss.")
    else:
        logger.debug(f"NoHub final loss = {losses[-1]}")

    no_hub.eval()
    embeddings = no_hub()
    return embeddings, losses  #为啥他加了detach
