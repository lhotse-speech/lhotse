"""
I'm thinking that for greater flexibility, we should have multiple layers of dataset rather than one unified notion of data-dir.
And while we should support the notion of a PyTorch dataset, these objects might not coincide with that- they would be
a more general way of storing information about a speech dataset, and where needed we can build a PyTorch
dataloader on top of them.  I'll call things Set instead of Dataset in order to make the distinction from a PyTorch notion of dataset.

Please treat this as just a proposal, that can be ripped up as well as modified.

RecordingSet: // I know this name has been taken already...
   - basically maps from string-valued recording-id -> the associated audio.
   - For each recording, you would be able to access: the number of channels, the sampling rate, the number of samples (and of course the length in seconds); and when needed you'd be able to access the audio (or at least, selected channels of it).  Of course the metadata would be in the manifest.
   - Where possible it would be nice to support getting a chunk of a selected subset of channels of the audio.  (i.e. have that in the interface, to account for cases where it's possible to seek in the data).
   - Bear in mind that there may be cases where different channels are stored in different files, like in AMI, and cases where we need a command to extract the data, so the interface should be general enough, e.g. don't enable extracting the filename.
   - When distributing data, like from OpenSLR, we may in some cases want to distribute the manifest file along with it, to avoid the user having to launch a time-consuming command, so it might be nice to support relative pathnames and setting a global root-directory prefix.

TextSupervisionSet:
   - This object would contain plaintext supervision information associated with audio.
   - Could be viewed as a list of segments within audio.
   - Could be viewed as a map from (utterance-id) -> (recording-id, start-time, end-time, text, ...), where "..." might include channel (in some scenarios), beamforming information in others.
   I'd rather not define in advance the set of possible fields.
   - The utterance-id could be automatically generated, maybe?  In fact, we might not always need an utterance-id.  But my feeling is it might come in useful, and it would make the interface more consistent with RecordingSet.

FeatureSet:
   - Represents a set of extracted features associated with recordings.
     (could also have dynamic versions).
   - Caution: we shouldn't assume that we are extracting features for the entire recording, or that the frame rates are always constant (perturbing the frame rates might be useful).
   - I don't want to obscure the relationship between segments and the original recordings by introducing another arbitrary level of id (like the 'cut' we discussed on the call, but see later).   What I am thinking is that the features would be accessible by recording-id, optional channel-info and time, maybe?   E.g. "what do you have for channel 0 of recording 'foo' between t=16 and t=22.2" ?
   - Again, we should make the metadata available separately from the actual data.
  - Please see the `lilcom` project on my github, which I have now finalized with the aim being to support compression of feature files in a general numpy-compatible way.  This will be useful here (but maybe shouldn't be visible from the interface).


 CutSet:
   - This would represent a collection of pieces of recordings that we can train a neural network on.  (For now let's assume
     the supervision is text; later on we'll have a stage of expansion where it's FSAs, but that stuff isn't finished
     yet so treat it as text for purposes of prototyping etc).
   - My notion of 'cut' is a little more general than of a typical segment for ASR training.  A 'cut' would consist
     of a matrix of features together with zero or more 'segments' where each segment has { a (begin, end)
     frame within the cut, and  the text sequence for the supervision}.    We can require that the segments be
     non-overlapping.  The idea is that within each segment we know what the
     supervision is supposed to be (it could even be the empty sequence).  Outside of a segment we don't
     know what the text sequence is supposed to be (e.g. it might not necessarily be silence).


The actual training would require us to form suitable minibatches from the CutSet.  I think rather than trying to
treat the CutSet as a PyTorch dataset (which I think has a too-limited interface for ASR), it will be better to
code the PyTorch dataloader ourselves, one that knows about the interface of CutSet.

  So, e.g. class CutSetDataloader(torch.Dataloader) : ....


 It might be a good idea to assign 'ids' to the cuts in the CutSet as well, even if they are just arbitrary hex strings.
 But others may have opinions on this.

Note: in general it will be possible to both truncate and append cuts, because of the way they have internal
segments and allow for padding with acoustic context (we might always create them with extra audio that we
can truncate).  We can even envision things like adding together the energies from two cuts, as long as their
segments don't overlap, to form training data suitable for multi-speaker conversations.
"""
