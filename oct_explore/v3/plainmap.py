"""Map a costgx-style shared vector x = L*[bre,bim,th,ph,bc,g] + [bre,bim,th,ph] onto v3core arrays."""
import numpy as np, jax.numpy as jnp
NP, NB = 6, 4
def nparams(L): return L * NP + NB
def split(x, L, C, kmax, B):
    X = x[: L * NP].reshape(L, NP)
    blocks = jnp.concatenate([X[:, :NB], x[L * NP:][None]], 0)            # (L+1,4)
    loc = jnp.broadcast_to(blocks[None, :, None, :], (B, L + 1, C, NB))
    mix = jnp.broadcast_to(X[:, 4][None, :, None, None], (B, L, C, kmax))
    return loc, mix, X[:, 5]
