import os
import logging
import json
from pathlib import Path
from tqdm import tqdm
import librosa

from speechbrain.utils.data_utils import get_all_files

from datasets.L2_ARCTIC.parse_textgrid import parse_textgrid

logger = logging.getLogger(__name__)

train_spks = [28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37, 2, 8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]
valid_spks = [26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]
test_spks = [28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]

train_spks = [f'{spk:02d}' for spk in train_spks]
valid_spks = [f'{spk:02d}' for spk in valid_spks]
test_spks = [f'{spk:02d}' for spk in test_spks]


def prepare(dataset_dir, train_json_path, valid_json_path, test_json_path, *args, **kwargs):
    # initialization
    dataset_dir = Path(dataset_dir)

    train_json_path = Path(train_json_path)
    valid_json_path = Path(valid_json_path)
    test_json_path = Path(test_json_path)
    set_names = ['train', 'valid', 'test']
    json_paths = [train_json_path, valid_json_path, test_json_path]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # check if this step is already performed, and skip it if so
    skip = True
    for json_path in json_paths:
        if not json_path.exists():
            skip = False
    if skip:
        logger.info('Skip preparation.')
        return

    logger.info('Generate json files.')

    # load forced alignment segmentation
    fa_seg_json_path = dataset_dir / 'forced_alignment_segmentation.json'
    with open(fa_seg_json_path) as f:
        fa_segmentation = json.load(f)

    # get speakers
    spk_lists = [train_spks, valid_spks, test_spks]
    json_data = []
    for set_name, json_path, spks in zip(set_names, json_paths, spk_lists):
        # match_and = ['/annotation/']
        # match_or = spks
        # ann_paths = get_all_files(
        #     dirName=dataset_dir,
        #     match_and=match_and,
        #     match_or=match_or
        # )
        # ann_paths = [Path(path) for path in ann_paths]
        metadata_paths = [dataset_dir / spk / f'{spk}_meta.json' for spk in spks]
        set_json_data = generate_json(json_path, metadata_paths, fa_segmentation)
        json_data.append(set_json_data)

    # prepare Kaldi files
    kaldi_root = os.environ.get('KALDI_ROOT')
    if kaldi_root is None:
        raise ValueError('KALDI_ROOT not set')
    kaldi_root = Path(kaldi_root)
    if not kaldi_root.is_dir():
        raise FileNotFoundError(f'KALDI_ROOT does not exist: {kaldi_root.absolute()}')
    # create Kaldi data directory
    kaldi_dir = dataset_dir.parent / 'kaldi_data'
    kaldi_dir.mkdir(exist_ok=True)

    for set_name, set_json_data in zip(set_names, json_data):
        # create wav.scp
        wav_scp_lines = []
        sr = 16000
        for utt_id, utt_data in set_json_data.items():
            wav_path = utt_data['wav_path']
            wav_path = Path(wav_path)
            assert wav_path.is_file()
            line = f'{utt_id} /usr/bin/sox {wav_path.absolute()} -t wav -r {sr} -b 16 - |\n'
            wav_scp_lines.append(line)
        wav_scp_path = kaldi_dir / f'{set_name}.wav.scp'
        with open(wav_scp_path, 'w') as f:
            f.writelines(wav_scp_lines)

        # create utt2spk
        utt2spk_lines = []
        for utt_id in set_json_data:
            spk = utt_id.split('_')[0]
            line = f'{utt_id} {spk}\n'
            utt2spk_lines.append(line)
        utt2spk_path = kaldi_dir / f'{set_name}.utt2spk'
        with open(utt2spk_path, 'w') as f:
            f.writelines(utt2spk_lines)


def generate_json(json_path, metadata_paths, fa_segmentation):
    json_data = {}

    for metadata_path in tqdm(sorted(metadata_paths)):
        with open(metadata_path) as f:
            spk_metadata = json.load(f)
        for utt_id, utt_metadata in spk_metadata.items():
            spk_id = metadata_path.parent.stem

            # wave file path
            wav_path = f'datasets/SynAudioMNIST/original_dataset/{spk_id}/{utt_id}.wav'

            # get duration
            duration = utt_metadata['duration']

            # canonical phonemes
            canonicals = utt_metadata['canonical_digit_seq']
            phonemes = utt_metadata['pronounced_digit_seq']
            gt_segments = utt_metadata['segment_seq']

            utt_fa_segmentation = fa_segmentation[utt_id]

            utt_json_data = {
                'wav_path': wav_path,
                'duration': duration,
                'spk_id': spk_id,
                'txt_gt_phn_seq': phonemes,
                'txt_gt_cnncl_seq': canonicals,
                'gt_seg_seq': gt_segments,
                'fa_seg_seq': utt_fa_segmentation
                # 'words': words,
                # 'word_segments': word_segments
            }
            json_data[utt_id] = utt_json_data

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    return json_data