import os
from pathlib import Path
import json

kaldi_root = os.environ.get('KALDI_ROOT')
if kaldi_root is None:
    raise ValueError('KALDI_ROOT not set')
kaldi_root = Path(kaldi_root)
if not kaldi_root.is_dir():
    raise FileNotFoundError(f'KALDI_ROOT does not exist: {kaldi_root.absolute()}')

dataset_name = 'L2_ARCTIC'

dataset_dir = Path('/home/weiwei/research/codes/MD-VAE-SpeechBrain/src/datasets')
dataset_dir = dataset_dir / dataset_name
annotation_dir = dataset_dir / 'annotation'
if not annotation_dir.is_dir():
    raise FileNotFoundError(f'directory does not exist: {annotation_dir.absolute()}')

# load annotation data from json files
json_data = {}
for set_name in ['train', 'valid', 'test']:
    set_json_file = annotation_dir / f'{set_name}.json'
    if not set_json_file.is_file():
        raise FileNotFoundError(f'file does not exist: {set_json_file.absolute()}')
    with open(set_json_file) as f:
        set_json_data = json.load(f)
    json_data.update(set_json_data)

# create Kaldi data directory
kaldi_dir = dataset_dir / 'kaldi_data'
kaldi_dir.mkdir(exist_ok=True)

# create wav.scp
wav_scp_lines = []
sph2pipe_path = kaldi_root / 'tools/sph2pipe_v2.5/sph2pipe'
if not sph2pipe_path.is_file():
    raise FileNotFoundError(f'sph2pipe not found at: {sph2pipe_path.absolute()}')
for utt_id, utt_data in json_data.items():
    wav_path = utt_data['wav_path']
    wav_path = dataset_dir.parent.parent / wav_path
    line = f'{utt_id} {sph2pipe_path.absolute()} -f wav {wav_path.absolute()} |\n'
    wav_scp_lines.append(line)

wav_scp_path = kaldi_dir / 'wav.scp'
with open(wav_scp_path, 'w') as f:
    f.writelines(wav_scp_lines)
