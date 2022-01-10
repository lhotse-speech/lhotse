from lhotse import close_cached_file_handles


class Hdf5MemoryIssueFix:
    """
    Use this class to limit the growing memory use when reading from HDF5 files.

    It should be instantiated withing the dataloading worker, i.e., the best place
    is likely inside the PyTorch Dataset class.

    Every time a new batch/example is returned, call ``.update()``.
    Once per ``reset_interval`` updates, this object will close all open HDF5 file
    handles, which seems to limit the memory use.
    """

    def __init__(self, reset_interval: int = 100) -> None:
        self.counter = 0
        self.reset_interval = reset_interval

    def update(self) -> None:
        if self.counter > 0 and self.counter % self.reset_interval == 0:
            close_cached_file_handles()
            self.counter = 0
        self.counter += 1
