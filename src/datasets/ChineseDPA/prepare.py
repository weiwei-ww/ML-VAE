import os
import json
import logging
from pathlib import Path
import tgt

logger = logging.getLogger(__name__)

def read_tg_file(path):
    tg_object = tgt.io.read_textgrid(path, include_empty_intervals=True)

    # read pinyin and segments
    segment_seq = []
    pinyin_seq = []
    pinyin_tier = tg_object.get_tier_by_name('initial/final')
    for interval in pinyin_tier:
        start_time = interval.start_time
        end_time = interval.end_time
        segment_seq.append([start_time, end_time])

        text = interval.text
        if text[-1].isdigit():
            text = text[:-1]
        pinyin_seq.append(text)

    # read misp labels
    misp_seq = []
    misp_tier = tg_object.get_tier_by_name('mispronunciation')
    for interval in misp_tier:
        text = interval.text
        if text == 'sil':
            text = ''
        # assert text in '+-*&', f'invalid misp label: {text}'
        if len(text) > 1:
            logger.warning(f'Convert misp label: {text} -> {text[0]}')
            text = text[0]
        misp_seq.append(text)
    return pinyin_seq, segment_seq, misp_seq


def prepare(dataset_dir, train_json_path, valid_json_path, test_json_path, *args, **kwargs):
    # initialization
    # dataset_dir = Path('datasets/ChineseDPA/original_dataset')
    dataset_dir = Path(dataset_dir)

    train_json_path = Path(train_json_path)
    valid_json_path = Path(valid_json_path)
    test_json_path = Path(test_json_path)
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

    # start preparation
    set_names = ['train', 'valid', 'test']
    output_json_paths = [train_json_path, valid_json_path, test_json_path]

    json_data = []
    for set_name, output_json_path in zip(set_names, output_json_paths):
        # load metadata
        metadata_file_name = f'metadata_{set_name}_spk_level.json'
        metadata_file_path = dataset_dir / metadata_file_name
        with open(metadata_file_path) as f:
            set_metadata = json.load(f)

        # generate json data
        set_json_data = {}
        for spk_id, spk_metadata in set_metadata.items():
            for utt_id, utt_metadata in spk_metadata.items():
                wav_path = dataset_dir / 'formatted_chinese_dpa' / utt_metadata['wav_path']
                duration = utt_metadata['duration']

                # parse textgrid file for canonical pinyin and forced alignment
                tg_file_path = wav_path.with_suffix('.TextGrid')
                txt_gt_cnncl_seq, fa_seg_seq, misp_lbl_seq = read_tg_file(tg_file_path)
                assert len(txt_gt_cnncl_seq) == len(fa_seg_seq) == len(misp_lbl_seq)

                # parse textgrid file for gt pinyin and gt segments (only for valid and test set)
                if set_name in ['valid', 'test']:
                    gt_tg_file_path = dataset_dir / 'human_annotation' / spk_id / f'{utt_id}.TextGrid'
                txt_gt_phn_seq, gt_seg_seq, misp_lbl_seq = read_tg_file(tg_file_path)
                assert len(txt_gt_phn_seq) == len(gt_seg_seq) == len(misp_lbl_seq)

                # handle mispronunciation labels and update gt_phn_seq with 'err'
                for i, misp_label in enumerate(misp_lbl_seq):
                    if misp_label != '':
                        txt_gt_phn_seq[i] = 'err'

                # # get canonical pinyin sequence
                # txt_gt_cnncl_seq = []
                # pinyin_seq = utt_metadata['ann_pinyin_seq']
                # for pinyin in pinyin_seq:
                #     initial = pinyin['initial']
                #     txt_gt_cnncl_seq.append(initial)
                #     if initial[-1].isdigit():
                #         initial = initial[:-1]
                #     final = pinyin['final']
                #     if final[-1].isdigit():
                #         final = final[:-1]
                #     txt_gt_cnncl_seq.append(final)
                utt_json_data = {
                    'wav_path': str(wav_path),
                    'duration': duration,
                    'spk_id': spk_id,
                    'txt_gt_phn_seq': txt_gt_phn_seq,
                    'txt_gt_cnncl_seq': txt_gt_cnncl_seq,
                    'gt_seg_seq': gt_seg_seq,
                    'fa_seg_seq': fa_seg_seq
                }
                set_json_data[utt_id] = utt_json_data

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(set_json_data, f, indent=4)
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
