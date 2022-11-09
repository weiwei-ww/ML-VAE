import json
from pathlib import Path
import tgt

dataset_path = Path('./original_dataset')

set_names = ['train', 'valid', 'test']

def read_tg_file(path):
    tg_object = tgt.io.read_textgrid(path)

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
        assert text in '+-*&', f'invalid misp label: {text}'
        misp_seq.append(text)
    return pinyin_seq, segment_seq, misp_seq

for set_name in set_names:
    # load metadata
    metadata_file_name = f'metadata_{set_name}_spk_level.json'
    metadata_file_path = dataset_path / metadata_file_name
    with open(metadata_file_path) as f:
        set_metadata = json.load(f)

    # generate json data
    set_json_data = {}
    for spk_id, spk_metadata in set_metadata:
        for utt_id, utt_metadata in spk_metadata.items():
            wav_path = 'formatted_chinese_dpa/' + utt_metadata['wav_path']
            duration = utt_metadata['duration']

            # parse textgrid file for canonical pinyin and forced alignment
            tg_file_path = Path(wav_path).with_suffix('.TextGrid')
            txt_gt_cnncl_seq, fa_seg_seq, misp_seq = read_tg_file(tg_file_path)

            # parse textgrid file for gt pinyin and gt segments (only for valid and test set)
            if set_name in ['valid', 'test']:
                gt_tg_file_path = dataset_path / 'human_annotation' / spk_id / f'{utt_id}.TextGrid'
            txt_gt_phn_seq, gt_seg_seq, misp_seq = read_tg_file(tg_file_path)

            # TODO: handle mispronunciation labels and update gt_phn_seq with 'err'

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
                'wav_path': wav_path,
                'duration': duration,
                'spk_id': spk_id,
                'txt_gt_phn_seq': txt_gt_phn_seq,
                'txt_gt_cnncl_seq': txt_gt_cnncl_seq,
                'gt_seg_seq': gt_seg_seq,
                'fa_seg_seq': fa_seg_seq
            }
            set_json_data[utt_id] = utt_json_data

