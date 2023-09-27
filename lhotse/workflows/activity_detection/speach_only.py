from lhotse.cut import CutSet


def speach_only(cutset: CutSet, root: str, num_jobs: int = 1) -> CutSet:
    # TODO: 1. Act on cutset elements, for each cut:
    # TODO: 1.1 Prepare Recording for speech analysis
    # TODO: 1.2 Analyze audio and select speech fragments
    # TODO: 1.3 Transform audio by removing silence according to selected fragments
    # TODO: 1.4 Save new audio to root
    # TODO: 1.4 Redefine Recording
    # TODO: 1.5 Transform supervision according to selected fragments
    # TODO: 1.7 Form a new cutset element
    # TODO: 2. Collect and return a new cutset
    # TODO: * Use multiprocessing to speed up the process
    # TODO: * Balance the load across processes
    # TODO: * Do not use more processes than the number of available CPUs
    # TODO: * Do not use more processes than the number of cuts
    # TODO: * Be careful not to overload the RAM
    # TODO: * Separate the cutset into chunks and process them separately?
    # TODO: * Save the new audio to disk during processing
    # TODO: * Make sure that the new audio is saved in the same format as the original audio
    # TODO: * Keep the original number of channels
    # TODO: * Keep the original sampling rate
    # TODO: * Drop supervisions that are not part of the new audio
    # TODO: * Keep the additional supervision information (e.g., speaker, language, etc.)
    # TODO: * Use tqdm to show progress
    return cutset
