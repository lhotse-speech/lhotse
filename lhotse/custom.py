from functools import partial
from typing import Any, Dict, Optional

import numpy as np

from lhotse import Recording
from lhotse.utils import asdict_nonull, fastcopy, ifnone


class CustomFieldMixin:
    """
    :class:`CustomFieldMixin` is intended for classes such as Cut or SupervisionSegment
    that support holding custom, user-defined fields.

    .. caution:: Due to the way inheritance and dataclasses work before Python 3.10,
        it is necessary to re-define ``custom`` attribute in dataclasses that
        inherit from this mixin.
    """

    def __init__(self, custom: Optional[Dict[str, Any]]) -> None:
        self.custom: Optional[Dict[str, Any]] = custom

    def __setattr__(self, key: str, value: Any) -> None:
        """
        This magic function is called when the user tries to set an attribute.
        We use it as syntactic sugar to store custom attributes in ``self.custom``
        field, so that they can be (de)serialized later.
        Setting a ``None`` value will remove the attribute from ``custom``.
        """
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            custom = ifnone(self.custom, {})
            if value is None:
                custom.pop(key, None)
            else:
                custom[key] = value
            if custom:
                self.custom = custom

    def __getattr__(self, name: str) -> Any:
        """
        This magic function is called when the user tries to access an attribute
        of :class:`.MonoCut` that doesn't exist. It is used for accessing the custom
        attributes of cuts.

        We use it to look up the ``custom`` field: when it's None or empty,
        we'll just raise AttributeError as usual.
        If ``item`` is found in ``custom``, we'll return ``custom[item]``.
        If ``item`` starts with "load_", we'll assume the name of the relevant
        attribute comes after that, and that value of that field is of type
        :class:`~lhotse.array.Array` or :class:`~lhotse.array.TemporalArray`.
        We'll return its ``load`` method to call by the user.

        Example of attaching and reading an alignment as TemporalArray::

            >>> cut = MonoCut('cut1', start=0, duration=4, channel=0)
            >>> cut.alignment = TemporalArray(...)
            >>> ali = cut.load_alignment()

        """
        custom = self.custom
        if custom is None:
            raise AttributeError(f"No such attribute: {name}")
        if name in custom:
            # Somebody accesses raw [Temporal]Array manifest
            # or wrote a custom piece of metadata into MonoCut.
            return self.custom[name]
        elif name.startswith("load_"):
            # Return the method for loading [Temporal]Arrays,
            # to be invoked by the user.
            attr_name = name[5:]
            return partial(self.load_custom, attr_name)
        raise AttributeError(f"No such attribute: {name}")

    def __delattr__(self, key: str) -> None:
        """Used to support ``del cut.custom_attr`` syntax."""
        if key in self.__dataclass_fields__:
            super().__delattr__(key)
        if self.custom is None or key not in self.custom:
            raise AttributeError(f"No such member: '{key}'")
        del self.custom[key]

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    def with_custom(self, name: str, value: Any):
        """Return a copy of this object with an extra custom field assigned to it."""
        cpy = fastcopy(
            self, custom=self.custom.copy() if self.custom is not None else {}
        )
        cpy.custom[name] = value
        return cpy

    def load_custom(self, name: str) -> np.ndarray:
        """
        Load custom data as numpy array. The custom data is expected to have
        been stored in cuts ``custom`` field as an :class:`~lhotse.array.Array` or
        :class:`~lhotse.array.TemporalArray` manifest.

        .. note:: It works with Array manifests stored via attribute assignments,
            e.g.: ``cut.my_custom_data = Array(...)``.

        :param name: name of the custom attribute.
        :return: a numpy array with the data.
        """
        from lhotse.array import Array, TemporalArray

        value = self.custom.get(name)
        if isinstance(value, Array):
            # Array does not support slicing.
            return value.load()
        elif isinstance(value, TemporalArray):
            # TemporalArray supports slicing.
            return value.load(start=self.start, duration=self.duration)
        elif isinstance(value, Recording):
            # Recording supports slicing.
            # Note: cut.channels referes to cut.recording and not the custom field.
            # We have to use a special channel selector field instead; e.g.:
            # if this is "target_recording", we'll look for "target_recording_channel_selector"
            channels = self.custom.get(f"{name}_channel_selector")
            return value.load_audio(
                channels=channels, offset=self.start, duration=self.duration
            )
        else:
            raise ValueError(
                f"To load {name}, the cut needs to have field {name} (or cut.custom['{name}']) "
                f"defined, and its value has to be a manifest of type Array or TemporalArray."
            )

    def has_custom(self, name: str) -> bool:
        """
        Check if the Cut has a custom attribute with name ``name``.

        :param name: name of the custom attribute.
        :return: a boolean.
        """
        if self.custom is None:
            return False
        return name in self.custom

    def drop_custom(self, name: str):
        if self.custom is None or name not in self.custom:
            return None
        del self.custom[name]
        return self
