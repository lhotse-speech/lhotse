from pathlib import Path

from speechcolab.datasets.gigaspeech import GigaSpeech

from lhotse.recipes.gigaspeech import prepare_gigaspeech

# Settings for paths
root_dir = Path('data')
corpus_dir = root_dir / 'GigaSpeech'
output_dir = root_dir / 'gigaspeech_nb'

# Select data parts
subset = '{XS}'

# Download the data
gigaspeech = GigaSpeech(corpus_dir)
# gigaspeech_data.download(subset)
assert gigaspeech.json_path.is_file()

# Prepare audio and supervision manifests
gigaspeech_manifests_train = prepare_gigaspeech(gigaspeech, '{XS}')
gigaspeech_manifests_test = prepare_gigaspeech(gigaspeech, '{TEST}')
