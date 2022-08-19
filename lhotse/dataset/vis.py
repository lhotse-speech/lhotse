from typing import Any, Mapping


def plot_batch(batch: Mapping[str, Any], supervisions: bool = True, text=True):
    import matplotlib.pyplot as plt

    batch_size = _get_one_of(batch, "features", "audio", "inputs").shape[0]
    fig, axes = plt.subplots(batch_size, figsize=(16, batch_size), sharex=True)

    def _plot_features(key):
        feats = batch[key]
        feat_actors = []
        for idx in range(batch_size):
            feat_actors.append(axes[idx].imshow(feats[idx].numpy().transpose()))
            axes[idx].invert_yaxis()
        fig.tight_layout(h_pad=2)
        fig.colorbar(feat_actors[-1], ax=axes)

    if "features" in batch:
        _plot_features(key="features")

    if "audio" in batch:
        raise NotImplementedError("Plotting audio for batches is not supported yet.")

    if "inputs" in batch:
        # For now, assume it is features and not multi-channel audio...
        if len(batch["inputs"].shape) == 3:
            _plot_features(key="inputs")
        else:
            raise NotImplementedError(
                "We could not infer what does the key 'inputs' represent yet."
            )

    if supervisions and "supervisions" in batch:
        sups = batch["supervisions"]
        for idx in range(len(sups["sequence_idx"])):
            seq_idx = sups["sequence_idx"][idx]
            if all(k in sups for k in ("start_frame", "num_frames")):
                start, end = (
                    sups["start_frame"][idx],
                    sups["start_frame"][idx] + sups["num_frames"][idx],
                )
            elif all(k in sups for k in ("start_sample", "num_samples")):
                start, end = (
                    sups["start_sample"][idx],
                    sups["start_sample"][idx] + sups["num_samples"][idx],
                )
            else:
                raise ValueError(
                    "Cannot plot supervisions: missing 'start_frame/sample' and 'num_frames/samples' fields."
                )
            axes[seq_idx].axvspan(
                start, end, fill=False, edgecolor="red", linestyle="--", linewidth=4
            )
            if text and "text" in sups:
                axes[seq_idx].text(start, 83, sups["text"][idx])


def _get_one_of(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
