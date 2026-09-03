"""Parallel-beam Radon transform, and its true adjoint.

This is the first brick of the whole constructive leg, so its acceptance criterion has to be
hard: **the adjoint test**

    <A x, y>  ==  <x, A^T y>      for arbitrary x, y

This is not a "might as well check it too" test. It is the entry certificate of the field: if
A^T is not the adjoint of A, then every step used in the derivation of every downstream gradient-
or duality-based solver (FISTA / ADMM / Chambolle-Pock) fails to hold, while the code **still
runs and still outputs something that looks like a reconstruction**. That is precisely the most
dangerous failure mode -- no error, real output, entirely wrong conclusions.

## A common but wrong approach

Many implementations write the "adjoint" as "backproject and interpolate back", or simply use
`skimage.transform.iradon` as the adjoint. `iradon` is **filtered backprojection**, an
approximate inverse of A rather than its adjoint; used as A^T, the adjoint test is off by orders
of magnitude. In this file `backproject` is the **unfiltered** backprojection, and that is the
adjoint of `project`.

## What the adjoint is (stepped on once here, written down)

The first version of this file wrote A^T as "rotate backwards and broadcast back over the
column", and the adjoint test immediately gave a relative error of 0.49-0.82 -- not an adjoint at
all. Root cause: A = S ∘ R (rotate first, then sum along columns), so A^T = R^T ∘ S^T. S^T
(broadcasting a row back over the column) was right; **R^T is not the reverse rotation**.
Bilinear `grid_sample` is a gather operator, and its adjoint is a scatter/splat; the two coincide
only when the sampling happens to be a permutation (integer multiples of 90 degrees).

**Lesson: the adjoint is not "the geometric inverse", it is the transpose of the linear map you
actually implemented.**

The right way to obtain it: A is linear (grid_sample on a fixed grid is linear, and so is the
sum), and in an autodiff framework the **VJP of a linear map is by definition its transpose**.
So A^T y is taken with `torch.autograd.grad`, exact to machine precision rather than approximate.
This also avoids the boundary and weight errors that are almost inevitable when writing the
scatter by hand.

Incidentally: many implementations use `skimage.transform.iradon` as the adjoint. `iradon` is
**filtered** backprojection, an approximate inverse of A rather than its adjoint, and using it
as the adjoint is off by orders of magnitude. `backproject` in this file is unfiltered.
"""
import torch
import torch.nn.functional as F

__all__ = ["RadonParallel"]


def _rotation_grid(n, angle, device, dtype):
    """The sampling grid needed to rotate an image by angle (radians), shape (1, n, n, 2)."""
    theta = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle),  torch.cos(angle), 0.0]],
        device=device, dtype=dtype).unsqueeze(0)
    return F.affine_grid(theta, (1, 1, n, n), align_corners=False)


class RadonParallel:
    """Parallel-beam Radon: image (n,n) -> sinogram (n_angles, n).

    angles  projection angles given in radians
    n       image side length (square)

    Convention: row i of the sinogram = the image rotated by -angles[i] and summed along columns.
    Summing along columns is a linear operator whose adjoint broadcasts each row back over the
    column, so the whole of A^T is "broadcast -> rotate back -> accumulate".
    """

    def __init__(self, n, angles, device="cpu", dtype=torch.float64):
        self.n = int(n)
        self.angles = torch.as_tensor(angles, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        # forward and backward share the same grids so the geometry stays consistent -- if each
        # side computed its own, the adjoint test would mismatch slightly, in a way that is very
        # hard to track down.
        self._fwd = [_rotation_grid(self.n, -a, device, dtype) for a in self.angles]
        self._bwd = [_rotation_grid(self.n, a, device, dtype) for a in self.angles]

    @property
    def in_shape(self):
        return (self.n, self.n)

    @property
    def out_shape(self):
        return (len(self.angles), self.n)

    def __call__(self, x):
        return self.project(x)

    def project(self, x):
        """A x: image -> sinogram."""
        x = x.reshape(1, 1, self.n, self.n).to(self.device, self.dtype)
        rows = []
        for g in self._fwd:
            rot = F.grid_sample(x, g, align_corners=False, padding_mode="zeros")
            rows.append(rot.sum(dim=2).reshape(-1))      # sum along columns
        return torch.stack(rows, dim=0)

    def backproject(self, y):
        """A^T y: sinogram -> image. The **unfiltered** backprojection, and a true adjoint.

        Taken via VJP: A is linear, so the vector-Jacobian product of project at any point equals
        A^T applied to that vector, independently of the expansion point. It is evaluated at x=0
        purely because that is cheap.
        """
        y = y.reshape(len(self.angles), self.n).to(self.device, self.dtype)
        x0 = torch.zeros(self.n, self.n, device=self.device, dtype=self.dtype,
                         requires_grad=True)
        out = self.project(x0)
        g, = torch.autograd.grad(out, x0, grad_outputs=y, create_graph=False)
        return g.reshape(self.n, self.n)


def adjoint_error(op, n_trials=8, seed=0, device="cpu", dtype=torch.float64):
    """Adjoint test: return the maximum of |<Ax,y> - <x,A^T y>| / |<Ax,y>|.

    Random x, y rather than special vectors: structured inputs may make the two sides agree by
    coincidence and hide a real mismatch.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    worst = 0.0
    for _ in range(n_trials):
        x = torch.randn(*op.in_shape, generator=g, dtype=dtype).to(device)
        y = torch.randn(*op.out_shape, generator=g, dtype=dtype).to(device)
        lhs = torch.sum(op.project(x) * y)
        rhs = torch.sum(x * op.backproject(y))
        denom = max(abs(float(lhs)), 1e-30)
        worst = max(worst, abs(float(lhs - rhs)) / denom)
    return worst
