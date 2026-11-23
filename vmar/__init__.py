"""vMAR's simple, scientific Python interface backed by the C23 core.

    from vmar import Differential

Only differential integration is promoted here. Neural learning, memory and
control retain their own C23 contracts and are not wrapped by this milestone.
"""

from .differential import Differential, Trajectory, VmarError

__all__ = ["Differential", "Trajectory", "VmarError"]
