import jax.numpy as jnp
import numpy as np

def _is_same_state(s1: jaxtyping.PyTree, s2: jaxtyping.PyTree) -> bool:
    """Returns whether two states refer to the same Params."""
    return np.all(

        jax.tree.map(

            lambda x, y: x is y,

            jax.tree_util.tree_leaves(s1),

            jax.tree_util.tree_leaves(s2),

        )

    ) 