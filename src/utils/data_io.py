import json
import logging
from tqdm import tqdm
import pickle
from pathlib import Path
import copy
import librosa
from kaldiio import ReadHelper
from joblib import Parallel, delayed

import torch
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline

from utils.data_io_utils import \
    generate_flvl_annotation, generate_boundary_seq, compute_fbank_kaldi, get_label_encoder, \
    apply_flvl_saved_md_results, apply_plvl_saved_md_results, apply_boundary_saved_md_results

logger = logging.getLogger(__name__)

output_keys = [
    'id',
    'wav', 'aug_wav',  # wave form
    'duration',  # duration
    'feat', 'aug_feat',  # feature
    'kaldi_feat', 'aug_kaldi_feat',  # Kaldi feature
    'gt_phn_seq', 'gt_cnncl_seq',  # encoded phonemes
    'flvl_gt_phn_seq', 'flvl_gt_cnncl_seq',  # frame level phonemes
    'aug_flvl_gt_cnncl_seq', 'aug_flvl_gt_cnncl_seq',  # frame level phoneme with augmentation
    'plvl_gt_md_lbl_seq', 'flvl_gt_md_lbl_seq', 'aug_flvl_gt_md_lbl_seq',  # phoneme and frame level MD ground truth
    'gt_seg_seq', 'gt_boundary_seq', 'gt_phn_end_seq',  # ground truth segmentation
    'fa_seg_seq', 'fa_boundary_seq', 'fa_phn_end_seq',  # forced alignment segmentation
    'prior',  # prior distribution for phonemes
]


def prepare_datasets(hparams):
    logger.info('Preparing datasets.')
    dataset_dir = Path(hparams['prepare']['dataset_dir']).parent
    computed_dataset_dir = dataset_dir / 'computed_dataset'

    # check if pre-computed datasets exist
    set_names = ['train', 'valid', 'test']
    to_prepare = False
    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        if not pkl_path.exists():  # prepare computed dataset
            to_prepare = True

    # prepare dataset or load pre-computed datasets
    computed_datasets = []
    if to_prepare:
        logger.info('One or more computed datasets do not exist, start preparing them.')

        # compute Kaldi features
        for set_name in set_names:
            wav_scp_path = dataset_dir / 'kaldi_data' / f'{set_name}.wav.scp'
            compute_fbank_kaldi(wav_scp_path, hparams['kaldi_feature_params'])

        # prepare datasets
        datasets = data_io_prep(hparams)

        # save datasets
        for set_name, dataset in zip(set_names, datasets):  # prepare dataset for train, valid, and test sets
            pkl_path = computed_dataset_dir / f'{set_name}.pkl'
            pkl_path.parent.mkdir(exist_ok=True)

            computed_dataset_dict = {}
            pbar = tqdm(dataset)
            pbar.set_description('computing dataset')
            for data_sample in pbar:  # for each data sample
                utt_id = data_sample['id']  # get ID
                data_sample_dict = {}  # data sample content as a dictionary
                for key in output_keys:
                    if key != 'id':
                        data_sample_dict[key] = copy.deepcopy(data_sample[key])  # use deep copy to prevent errors
                computed_dataset_dict[utt_id] = data_sample_dict

            # def compute(idx):
            #     data_sample = dataset[idx]
            #     utt_id = data_sample['id']  # get ID
            #     data_sample_dict = {}  # data sample content as a dictionary
            #     for key in output_keys:
            #         if key != 'id':
            #             data_sample_dict[key] = copy.deepcopy(data_sample[key])  # use deep copy to prevent errors
            #     return utt_id, data_sample_dict
            # compute_datasets_result_list = Parallel(n_jobs=16)(delayed(compute)(idx) for idx in tqdm(range(len(dataset))))
            # computed_dataset_dict = {}
            # for utt_id, data_sample_dict in compute_datasets_result_list:
            #     computed_dataset_dict[utt_id] = data_sample_dict

            # save the computed dataset
            with open(pkl_path, 'wb') as f:
                pickle.dump(computed_dataset_dict, f)
    else:
        logger.info('Load pre-computed datasets.')

    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        with open(pkl_path, 'rb') as f:
            computed_dataset_dict = pickle.load(f)

        # apply saved MD results as a data cleaning step
        if hparams.get('apply_saved_md_results', False):
            json_dir = Path('datasets') / hparams['dataset'] / 'saved_md_results'
            json_path = json_dir / (hparams['saved_md_results_model_name'] + '.json')
            with open(json_path) as f:
                saved_md_results = json.load(f)

            logger.info('Apply saved MD results.')
            pbar = tqdm(computed_dataset_dict)
            pbar.set_description('applying saved MD results')
            for utt_id in pbar:
                flvl_len = len(computed_dataset_dict[utt_id]['feat'])
                plvl_len = len(computed_dataset_dict[utt_id]['gt_phn_seq'])
                for key, data in computed_dataset_dict[utt_id].items():
                    if 'flvl_' in key or 'feat' in key:
                        assert len(data) == flvl_len
                        updated_data = apply_flvl_saved_md_results(data, saved_md_results[utt_id])
                    elif 'boundary_' in key:
                        assert len(data) == flvl_len
                        updated_data = apply_boundary_saved_md_results(data, saved_md_results[utt_id])
                    elif '_seq' in key:
                        assert len(data) == plvl_len
                        updated_data = apply_plvl_saved_md_results(data, saved_md_results[utt_id])
                    else:
                        updated_data = data
                    computed_dataset_dict[utt_id][key] = updated_data

                # test


        computed_dataset = sb.dataio.dataset.DynamicItemDataset(computed_dataset_dict, output_keys=output_keys)
        computed_datasets.append(computed_dataset)

    label_encoder = get_label_encoder(hparams)

    # save label_encoder
    label_encoder_path = computed_dataset_dir / 'label_encoder.txt'
    label_encoder.save(label_encoder_path)


    return computed_datasets, label_encoder


def data_io_prep(hparams):
    # datasets initialization
    def dataset_prep(hparams, set_name):
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams['prepare'][f'{set_name}_json_path'])

        if hparams['sorting'] in ['ascending', 'descending']:
            reverse = True if hparams['sorting'] == 'descending' else False
            dataset = dataset.filtered_sorted(sort_key='duration', reverse=reverse)
            hparams['train_dataloader_opts']['shuffle'] = False

        return dataset

    set_names = ['train', 'valid', 'test']
    train_dataset = dataset_prep(hparams, 'train')
    valid_dataset = dataset_prep(hparams, 'valid')
    test_dataset = dataset_prep(hparams, 'test')
    datasets = [train_dataset, valid_dataset, test_dataset]

    label_encoder = get_label_encoder(hparams)

    # Kaldi feature pipelines
    kaldi_feats = {}
    for set_name in set_names:
        feats_scp_path = Path(hparams['prepare']['dataset_dir']).parent / 'kaldi_data' / f'{set_name}.feats.scp'
        with ReadHelper(f'scp:{feats_scp_path.absolute()}') as reader:
            for utt_id, feat in reader:
                kaldi_feats[utt_id] = torch.from_numpy(feat.copy())

    @speechbrain.utils.data_pipeline.takes('id')
    @speechbrain.utils.data_pipeline.provides('kaldi_feat', 'aug_kaldi_feat')
    def kaldi_feat_pipeline(id):
        yield kaldi_feats[id]
        yield kaldi_feats[id]

    sb.dataio.dataset.add_dynamic_item(datasets, kaldi_feat_pipeline)

    # audio pipelines
    @speechbrain.utils.data_pipeline.takes('wav_path', 'kaldi_feat')
    @speechbrain.utils.data_pipeline.provides('wav', 'feat', 'aug_wav', 'aug_feat')
    def audio_pipeline(wav_path, kaldi_feat):
        # use librosa instead of sb to get the correct sample rate
        sr = hparams['sample_rate']
        wav, _ = librosa.load(wav_path, sr=sr)
        wav = torch.from_numpy(wav)
        # wav = speechbrain.dataio.dataio.read_audio(wav_path)
        yield wav  # wave form

        batched_wav = wav.unsqueeze(dim=0)  # add a batch dimension
        feat = hparams['compute_features'](batched_wav).squeeze(dim=0)
        if feat.shape[0] != kaldi_feat.shape[0]:
            assert feat.shape[0] - kaldi_feat.shape[0] == 1
            feat = feat[:kaldi_feat.shape[0], :]
        yield feat  # feature

        aug_wav = wav
        if hparams.get('augmentation'):
            aug_wav = hparams['augmentation'](batched_wav, torch.ones(1)).squeeze(dim=0)
        yield aug_wav  # wave form with augmentation

        batched_aug_wav = aug_wav.unsqueeze(dim=0)
        aug_feat = hparams['compute_features'](batched_aug_wav).squeeze(dim=0)
        if aug_feat.shape[0] != kaldi_feat.shape[0]:
            assert aug_feat.shape[0] - kaldi_feat.shape[0] == 1
            aug_feat = aug_feat[:kaldi_feat.shape[0], :]
        yield aug_feat  # feature with augmentation

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # text pipelines
    # frame level phonemes
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_seg_seq', 'txt_gt_phn_seq')
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
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_seg_seq', 'txt_gt_cnncl_seq')
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
    @speechbrain.utils.data_pipeline.takes('flvl_gt_phn_seq', 'flvl_gt_cnncl_seq', 'aug_flvl_gt_phn_seq',
                                           'aug_flvl_gt_cnncl_seq')
    @speechbrain.utils.data_pipeline.provides('flvl_gt_md_lbl_seq', 'aug_flvl_gt_md_lbl_seq')
    def flvl_gt_md_lbl_seq_pipeline(flvl_gt_phn_seq, flvl_gt_cnncl_seq, aug_flvl_gt_phn_seq, aug_flvl_gt_cnncl_seq):
        flvl_gt_md_lbl_seq = torch.ne(flvl_gt_phn_seq, flvl_gt_cnncl_seq).long()
        yield flvl_gt_md_lbl_seq
        aug_flvl_gt_md_lbl_seq = torch.ne(aug_flvl_gt_phn_seq, aug_flvl_gt_cnncl_seq).long()
        yield aug_flvl_gt_md_lbl_seq

    sb.dataio.dataset.add_dynamic_item(datasets, flvl_gt_md_lbl_seq_pipeline)

    # ground truth boundaries
    @speechbrain.utils.data_pipeline.takes('id', 'feat', 'duration', 'gt_seg_seq')
    @speechbrain.utils.data_pipeline.provides('gt_boundary_seq', 'gt_phn_end_seq')
    def gt_boundary_seq_pipeline(id, feat, duration, gt_segmentation):
        boundary_seq, phn_end_seq = generate_boundary_seq(id, feat, duration, gt_segmentation)
        yield boundary_seq
        yield phn_end_seq

    sb.dataio.dataset.add_dynamic_item(datasets, gt_boundary_seq_pipeline)

    # forced alignment boundaries
    @speechbrain.utils.data_pipeline.takes('id', 'feat', 'duration', 'fa_seg_seq')
    @speechbrain.utils.data_pipeline.provides('fa_boundary_seq', 'fa_phn_end_seq')
    def fa_boundary_seq_pipeline(id, feat, duration, fa_segmentation):
        boundary_seq, phn_end_seq = generate_boundary_seq(id, feat, duration, fa_segmentation)
        yield boundary_seq
        yield phn_end_seq

    sb.dataio.dataset.add_dynamic_item(datasets, fa_boundary_seq_pipeline)

    # set output keys
    sb.dataio.dataset.set_output_keys(datasets, ['gt_cnncl_seq'])

    # compute prior distribution
    logger.info('Compute prior for each phoneme.')
    prior = torch.zeros(len(label_encoder))
    def get_cnncl_phn_seq(train_sample_index):
        train_sample = train_dataset[train_sample_index]
        cnncl_phn_seq = train_sample['gt_cnncl_seq']
        return cnncl_phn_seq
    # pbar = tqdm(range(len(train_dataset)))
    # pbar.set_description('computing prior')
    # cnncl_phn_seqs = Parallel(n_jobs=1)(delayed(get_cnncl_phn_seq)(train_sample_index) for train_sample_index in pbar)
    # for cnncl_phn_seq in cnncl_phn_seqs:
    #     for cnncl_phn in cnncl_phn_seq:
    #         prior[cnncl_phn] += 1
    # prior /= torch.sum(prior)

    prior = torch.zeros(len(label_encoder))
    pbar = tqdm(train_dataset)
    pbar.set_description('computing prior')
    for train_sample in pbar:
        cnncl_phn_seq = train_sample['gt_cnncl_seq']
        for cnncl_phn in cnncl_phn_seq:
            prior[cnncl_phn] += 1
    prior /= torch.sum(prior)

    @speechbrain.utils.data_pipeline.provides('prior')
    def prior_pipeline():
        return prior

    sb.dataio.dataset.add_dynamic_item(datasets, prior_pipeline)

    # set output keys
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_dataset, valid_dataset, test_dataset
