"""Adapter.

Provides an adapter between packages
"""


class Adapter:
    """Adapter Class.

    Returns
    -------
    Object : Adapter
        Adapted Object

    """

    def __init__(self, obj, **adapted_methods):
        """."""
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        """All non-adapted methods pass through."""
        return getattr(self.obj, attr)

    def original_dict(self):
        """."""
        return self.obj.__dict__
