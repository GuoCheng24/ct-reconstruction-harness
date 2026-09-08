"""Learned Primal-Dual (Adler & Öktem 2018) -- the official dival architecture, wired to
this repository's operators so that the operator can be swapped after training.

## Why the official architecture and not a re-implementation

"A baseline you cannot beat is a recipe you have not finished reading." The LoDoPaB
leaderboard's LPD at 36.25 is the number this project has to talk to, and any home-grown
approximation of the architecture turns "our method wins" into "we beat a crippled LPD".
dival 0.6.2's PrimalDualNet is the code that produced 36.25; it is instantiated directly.

Official hyper-parameters (supp.dival, lodopab_learnedpd_hyper_params.json, checked):
    niter=10, internal_ch=64, lr=1e-4, batch_size=1, epochs=10,
    init_fbp=True, init_frequency_scaling=0.7

## Recipe archaeology: five defaults that live only in the source, not in the JSON

The first version of this file matched the JSON and still differed from the official model
in five places, all defaults of HYPER_PARAMS in learnedpd_reconstructor.py. In order of
damage:

  1. batch_norm defaults to False (the first version used True). Fatal: with the official
     batch_size=1, BatchNorm normalizes each image by its own statistics and erases the
     amplitude information the reconstruction lives on.
  2. prelu defaults to True (the first version used LeakyReLU 0.2).
  3. n_layer defaults to 3 (the first version used 4; 877,800 vs 1,618,920 parameters).
  4. Xavier-uniform initialization with zero bias (learnedpd_reconstructor.py:139-142;
     the first version used PyTorch's default Kaiming).
  5. normalize_by_opnorm=True: the network uses A/||A|| internally and the observation
     is divided by ||A|| during training, while the FBP initialization is multiplied back
     by ||A|| and so sees the raw y. ||A|| = 0.888 here, a 12% scale change; least harmful.

Controlled comparison, same data, same step 2000: the wrong recipe reaches 13.02 dB, the
official one 35.73 dB. How it was noticed: on the same 3522 training pairs, FBP+U-Net
reached 36.16 after 6000 steps (about the official U-Net 36.00), while LPD after 35220
steps sat at 33.09 -- below classical TV at 33.30. If the bottleneck were data, the U-Net
would also be 3 dB short. It was not. The quickest way to tell "not enough data" from
"I implemented it wrong" is whether another architecture reaches its own published number
on the same data.

## The one addition here: the operator is swappable

PrimalDualNet calls self.op / self.op_adj / self.op_init explicitly in its forward pass
(iterative.py:109/115/104), so "swap the operator at deployment" is one line:
    net.set_operator(new_op)
That is the structural fact the mismatch experiments rest on: an unrolled method keeps the
operator in the architecture (swappable), a post-processing method (FBP+U-Net) bakes it
into the initialization (not swappable).
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

IMG_SHAPE = (362, 362)
SINO_SHAPE = (1000, 513)


class _OpAdapter(nn.Module):
    """Wrap a MismatchedOperator method as the module PrimalDualNet expects.

    PrimalDualNet only uses `.operator.domain.shape` to allocate tensors, plus forward.
    Inputs arrive as (B, C, ...); the channel axis is flattened and restored here, because
    odl's OperatorModule tolerates leading axes but our own shift composition does not.
    """

    def __init__(self, fn, domain_shape, out_shape):
        super().__init__()
        self.fn = fn
        self.out_shape = out_shape
        self.operator = SimpleNamespace(domain=SimpleNamespace(shape=domain_shape))

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        out = self.fn(x.reshape(b * c, *x.shape[2:]))
        return out.reshape(b, c, *self.out_shape)


class LPD(nn.Module):
    """Learned Primal-Dual with a swappable operator."""

    def __init__(self, operator, n_iter=10, internal_ch=64, n_primal=5, n_dual=5,
                 init_fs=0.7, n_layer=3, normalize_by_opnorm=True,
                 batch_norm=False, prelu=True, device="cuda:0"):
        super().__init__()
        import torch.nn as nn
        from dival.reconstructors.networks.iterative import PrimalDualNet
        self.device = device
        self.n_iter, self.internal_ch = n_iter, internal_ch
        self.n_primal, self.n_dual, self.init_fs = n_primal, n_dual, init_fs
        self.normalize_by_opnorm = normalize_by_opnorm
        self.net = PrimalDualNet(
            n_iter=n_iter,
            op=_OpAdapter(lambda x: x, IMG_SHAPE, SINO_SHAPE),
            op_adj=_OpAdapter(lambda y: y, SINO_SHAPE, IMG_SHAPE),
            op_init=None, n_primal=n_primal, n_dual=n_dual,
            use_sigmoid=False, n_layer=n_layer, internal_ch=internal_ch,
            kernel_size=3, batch_norm=batch_norm, prelu=prelu, lrelu_coeff=0.2,
        ).to(device)

        # Official initialization (learnedpd_reconstructor.py:139-142): zero bias + Xavier
        def _init(m):
            if isinstance(m, nn.Conv2d):
                m.bias.data.fill_(0.0)
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(_init)
        self.set_operator(operator)

    def set_operator(self, operator, init_fs=None):
        """Swap the operator -- the entry point of the deployment-side mismatch experiments.
        The weights are not touched.

        Normalization follows the official convention: the network uses A/||A|| and
        (A/||A||)^T internally, and forward() divides the input y by ||A||; op_init multiplies
        by ||A|| first and then applies FBP, so it sees the raw y.
        """
        import odl
        fs = self.init_fs if init_fs is None else init_fs
        self.operator = operator
        self.opnorm = (float(odl.power_method_opnorm(operator.rt))
                       if self.normalize_by_opnorm else 1.0)
        s = self.opnorm
        fbp = operator.make_fbp(fs)
        self.net.op = _OpAdapter(lambda x: operator.project(x) / s,
                                 IMG_SHAPE, SINO_SHAPE).to(self.device)
        self.net.op_adj = _OpAdapter(lambda y: operator.adjoint(y) / s,
                                     SINO_SHAPE, IMG_SHAPE).to(self.device)
        self.net.op_init = _OpAdapter(lambda y: fbp(y * s),
                                      SINO_SHAPE, IMG_SHAPE).to(self.device)
        return self

    def forward(self, y):
        """y: (B,1000,513) -> (B,362,362). The input is divided by ||A|| per the official convention."""
        return self.net((y / self.opnorm)[:, None])[:, 0]

    def reconstruct(self, obs):
        """The harness method interface: method(obs) -> recon, one image."""
        self.eval()
        with torch.no_grad():
            return self(obs[None].to(self.device))[0]

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
