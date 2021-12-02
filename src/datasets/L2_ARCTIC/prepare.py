import logging
import json
from pathlib import Path
from tqdm import tqdm

import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files

from datasets.L2_ARCTIC.parse_textgrid import parse_textgrid

logger = logging.getLogger(__name__)

train_spks = '''ABA
ASI
BWC
EBVS
ERMS
HKK
HQTV
LXC
MBMPS
NCC
RRBI
SKA
SVBI
THV
YBAA'''

valid_spks = '''HJK
PNV
YDCK'''

test_spks = '''NJS
TLV
TNI
TXHC
YKWK
ZHAA'''

def prepare(dataset_dir, train_json_path, valid_json_path, test_json_path, phoneme_set_handler):
    # initialization
    dataset_dir = Path(dataset_dir)

    train_json_path = Path(train_json_path)
    valid_json_path = Path(valid_json_path)
    test_json_path = Path(test_json_path)
    json_paths = [train_json_path, valid_json_path, test_json_path]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # check if this step is already performed, and skip it if so
    skip = True
    for json_path in [train_json_path, valid_json_path, test_json_path]:
        if not json_path.exists():
            skip = False
    if skip:
        logger.info('skip preparation')
        return

    logger.info('generate json files')

    # load forced alignment segmentation
    fa_seg_json_path = dataset_dir.parent / 'fa_segmentation.json'
    with open(fa_seg_json_path) as f:
        fa_segmentation = json.load(f)

    # get speakers
    spk_lists = [train_spks.split(), valid_spks.split(), test_spks.split()]

    for json_path, spks in zip(json_paths, spk_lists):
        match_and = ['/annotation/']
        match_or = spks
        ann_paths = get_all_files(
            dirName=dataset_dir,
            match_and=match_and,
            match_or=match_or
        )
        ann_paths = [Path(path) for path in ann_paths]
        generate_json(json_path, ann_paths, phoneme_set_handler, fa_segmentation)


def generate_json(json_path, ann_paths, phoneme_set_handler, fa_segmentation):
    json_data = {}

    for ann_path in tqdm(sorted(ann_paths)):
        spk_id = ann_path.parent.parent.stem
        utt_id = '{}_{}'.format(spk_id, ann_path.stem.split('_')[-1])

        # wave file path
        wav_path = f'datasets/L2_ARCTIC/original_dataset/{spk_id}/wav/{ann_path.stem}.wav'

        # parse the TextGrid annotation file
        parsed_tg = parse_textgrid(ann_path, 'all')
        duration = parsed_tg['end_time'] - parsed_tg['start_time']

        # canonical phonemes
        canonicals = []
        for _, _, p in parsed_tg['canonical_phoneme']:
            canonicals.append(phoneme_set_handler.map_phoneme(p))

        # pronounced phoneme and segments
        phonemes = []
        phoneme_segments = []
        for start_time, end_time, p in parsed_tg['phoneme']:
            phonemes.append(phoneme_set_handler.map_phoneme(p))
            phoneme_segments.append([start_time, end_time])

        # words
        words = []
        word_segments = []
        for start_time, end_time, w in parsed_tg['word']:
            words.append(w)
            word_segments.append([start_time, end_time])

        utt_fa_segmentation = fa_segmentation[utt_id]

        utt_json_data = {
            'wav_path': wav_path,
            'duration': duration,
            'spk_id': spk_id,
            'txt_gt_phn_seq': phonemes,
            'txt_gt_cnncl_seq': canonicals,
            'gt_segmentation': phoneme_segments,
            'fa_segmentation': utt_fa_segmentation
            # 'words': words,
            # 'word_segments': word_segments
        }
        json_data[utt_id] = utt_json_data

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)