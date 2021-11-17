import logging
from tqdm import tqdm
import pickle
from pathlib import Path
import librosa

import torch
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline

from utils.preprocessing import generate_flvl_annotation

logger = logging.getLogger(__name__)


def prepare_datasets(hparams):
    logger.info('Preparing datasets.')
    datasets, label_encoder = data_io_prep(hparams)

    logger.info('Preparing computed datasets.')
    computed_datasets = []
    set_names = ['train', 'valid', 'test']
    output_keys = ['id']
    for set_name, dataset in zip(set_names, datasets):
        pkl_path = Path(hparams['prepare']['dataset_dir']).parent / 'computed_dataset' / f'{set_name}.pkl'
        if not pkl_path.exists():
            logger.info(f'Computed dataset for {set_name} set does not exist, start preparing it')
            pkl_path.parent.mkdir(exist_ok=True)
            computed_dataset_dict = {}
            for data_sample in tqdm(dataset):
                data_id = data_sample['id']
                data_sample_dict = {}
                for key in data_sample:
                    if key != 'id':
                        data_sample_dict[key] = data_sample[key]
                computed_dataset_dict[data_id] = data_sample_dict
            with open(pkl_path, 'wb') as f:
                pickle.dump(computed_dataset_dict, f)
        else:
            logger.info(f'Load computed dataset for {set_name} set')
            with open(pkl_path, 'rb') as f:
                computed_dataset_dict = pickle.load(f)

        if len(output_keys) == 1:
            output_keys.extend(computed_dataset_dict[list(computed_dataset_dict.keys())[0]].keys())

        computed_dataset = sb.dataio.dataset.DynamicItemDataset(computed_dataset_dict, output_keys=output_keys)
        computed_datasets.append(computed_dataset)

    test = computed_datasets[0][0]
    # for dataset in computed_datasets:
    #     for data_sample in dataset:
    #         n_frames = data_sample['feat'].shape[0]
    #         for key in data_sample:
    #             if key.startswith('flvl_'):
    #                 print(key, data_sample[key].shape[0])
    #                 assert data_sample[key].shape[0] == n_frames

    return computed_datasets, label_encoder


def data_io_prep(hparams):
    # datasets definition
    def dataset_prep(hparams, set_name):
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams['prepare'][f'{set_name}_json_path'])

        if hparams['sorting'] in ['ascending', 'descending']:
            reverse = True if hparams['sorting'] == 'descending' else False
            dataset = dataset.filtered_sorted(sort_key='duration', reverse=reverse)
            hparams['train_dataloader_opts']['shuffle'] = False

        return dataset

    train_dataset = dataset_prep(hparams, 'train')
    valid_dataset = dataset_prep(hparams, 'valid')
    test_dataset = dataset_prep(hparams, 'test')
    datasets = [train_dataset, valid_dataset, test_dataset]

    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # audio pipelines
    @speechbrain.utils.data_pipeline.takes('wav_path')
    @speechbrain.utils.data_pipeline.provides('wav', 'feat', 'aug_wav', 'aug_feat')
    def audio_pipeline(wav_path):
        # use librosa instead of sb to get the correct sample rate
        sr = hparams['sample_rate']
        wav, _ = librosa.load(wav_path, sr=sr)
        wav = torch.from_numpy(wav)
        # wav = speechbrain.dataio.dataio.read_audio(wav_path)
        yield wav  # wave form

        batched_wav = wav.unsqueeze(dim=0)  # add a batch dimension
        feat = hparams['compute_features'](batched_wav).squeeze(dim=0)
        # print(feat.shape)
        yield feat  # feature

        aug_wav = wav
        if hparams.get('augmentation'):
            aug_wav = hparams['augmentation'](batched_wav, torch.ones(1)).squeeze(dim=0)
        yield aug_wav  # wave form with augmentation

        batched_aug_wav = aug_wav.unsqueeze(dim=0)
        aug_feat = hparams['compute_features'](batched_aug_wav).squeeze(dim=0)
        yield aug_feat  # feature with augmentation

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # text pipelines
    # frame level phonemes
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_segmentation', 'txt_gt_phn_seq')
    @speechbrain.utils.data_pipeline.provides('gt_phn_seq', 'flvl_gt_phn_seq', 'aug_flvl_gt_phn_seq')
    def flvl_phoneme_pipeline(feat, aug_feat, duration, segmentation, txt_gt_phn_seq):
        gt_phn_seq = label_encoder.encode_sequence_torch(txt_gt_phn_seq)
        yield gt_phn_seq  # encoded phonemes
        flvl_gt_phn_seq = generate_flvl_annotation(label_encoder, feat, duration, segmentation, gt_phn_seq)
        yield flvl_gt_phn_seq  # frame level phonemes
        aug_flvl_gt_phn_seq = generate_flvl_annotation(label_encoder, aug_feat, duration, segmentation, gt_phn_seq)
        yield aug_flvl_gt_phn_seq  # frame level phonemes with augmentation
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_phoneme_pipeline)

    # frame level canonicals
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_segmentation', 'txt_gt_cnncl_seq')
    @speechbrain.utils.data_pipeline.provides('gt_cnncl_seq', 'flvl_gt_cnncl_seq', 'aug_flvl_gt_cnncl_seq')
    def flvl_canonical_pipeline(feat, aug_feat, duration, segmentation, txt_gt_cnncl_seq):
        gt_cnncl_seq = label_encoder.encode_sequence_torch(txt_gt_cnncl_seq)
        yield gt_cnncl_seq  # encoded canonicals
        flvl_gt_cnncl_seq = generate_flvl_annotation(label_encoder, feat, duration, segmentation, gt_cnncl_seq)
        yield flvl_gt_cnncl_seq  # frame level canonicals
        aug_flvl_gt_cnncl_seq = generate_flvl_annotation(label_encoder, aug_feat, duration, segmentation, gt_cnncl_seq)
        yield aug_flvl_gt_cnncl_seq  # frame level canonicals with augmentation
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_canonical_pipeline)

    # MD ground truth pipelines
    # phoneme level ground truth
    @speechbrain.utils.data_pipeline.takes('gt_phn_seq', 'gt_cnncl_seq')
    @speechbrain.utils.data_pipeline.provides('plvl_gt_md_lbl_seq')
    def plvl_gt_md_lbl_seq_pipeline(gt_phn_seq, gt_cnncl_seq):
        return torch.ne(gt_phn_seq, gt_cnncl_seq).long()
    sb.dataio.dataset.add_dynamic_item(datasets, plvl_gt_md_lbl_seq_pipeline)

    # frame level ground truth
    @speechbrain.utils.data_pipeline.takes('flvl_gt_phn_seq', 'flvl_gt_cnncl_seq', 'aug_flvl_gt_phn_seq', 'aug_flvl_gt_cnncl_seq')
    @speechbrain.utils.data_pipeline.provides('flvl_gt_md_lbl_seq', 'aug_flvl_gt_md_lbl_seq')
    def flvl_gt_md_lbl_seq_pipeline(flvl_gt_phn_seq, flvl_gt_cnncl_seq, aug_flvl_gt_phn_seq, aug_flvl_gt_cnncl_seq):
        flvl_gt_md_lbl_seq = torch.ne(flvl_gt_phn_seq, flvl_gt_cnncl_seq).long()
        yield flvl_gt_md_lbl_seq
        aug_flvl_gt_md_lbl_seq = torch.ne(aug_flvl_gt_phn_seq, aug_flvl_gt_cnncl_seq).long()
        yield aug_flvl_gt_md_lbl_seq
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_gt_md_lbl_seq_pipeline)

    # set output keys
    output_keys = [
        'id',
        'wav', 'aug_wav',  # wave form
        'feat', 'aug_feat',  # feature
        'gt_phn_seq', 'gt_cnncl_seq',  # encoded phonemes
        'flvl_gt_phn_seq', 'flvl_gt_cnncl_seq',  # frame level phonemes
        'aug_flvl_gt_cnncl_seq', 'aug_flvl_gt_cnncl_seq',  # frame level phoneme with augmentation
        'plvl_gt_md_lbl_seq', 'flvl_gt_md_lbl_seq', 'aug_flvl_gt_md_lbl_seq',  # phoneme and frame level MD ground truth
        'gt_segmentation'  # ground truth segmentation
    ]
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    # fit the label encoder
    phoneme_set = hparams['prepare']['phoneme_set_handler'].get_phoneme_set()
    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    label_encoder.insert_blank(index=hparams['blank_index'])

    test = datasets[0][0]
    # for key in test:
    #     print('-' * 500)
    #     print(key)
    #     print(test[key])

    return (train_dataset, valid_dataset, test_dataset), label_encoder