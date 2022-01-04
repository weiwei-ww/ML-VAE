import logging
from tqdm import tqdm
import pickle
from pathlib import Path
import copy
import librosa

import torch
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline

logger = logging.getLogger(__name__)

output_keys = [
    'id',
    'wav', 'aug_wav',  # wave form
    'duration',  # duration
    'feat', 'aug_feat',  # feature
    'gt_phn_seq', 'gt_cnncl_seq',  # encoded phonemes
    'flvl_gt_phn_seq', 'flvl_gt_cnncl_seq',  # frame level phonemes
    'aug_flvl_gt_cnncl_seq', 'aug_flvl_gt_cnncl_seq',  # frame level phoneme with augmentation
    'plvl_gt_md_lbl_seq', 'flvl_gt_md_lbl_seq', 'aug_flvl_gt_md_lbl_seq',  # phoneme and frame level MD ground truth
    'gt_seg_seq',  'gt_boundary_seq', 'gt_phn_end_seq',  # ground truth segmentation
    'fa_seg_seq', 'fa_boundary_seq', 'fa_phn_end_seq',  # forced alignment segmentation
    'prior'  # prior distribution for phonemes
]

def generate_flvl_annotation(label_encoder, feat, duration, segmentation, phoneme_list):
    """
    Generate frame level annotation.

    Parameters
    ----------
    label_encoder : sb.dataio.encoder.CTCTextEncoder
        label encoder
    feat : torch.Tensor
        (T, C), features
    duration : float
        total duration (in second)
    segmentation : list
        len = L, a list of tuples (start_second, end_second)
    phoneme_list : torch.LongTensor
        (L), list of phoneme level encoded phonemes

    Returns
    -------
    fl_phoneme_list : torch.Tensor
        T, list of frame level encoded phonemes
    """
    T = feat.shape[0]
    L = phoneme_list.shape[0]
    assert len(segmentation) == L

    # initialize the output with sil
    fl_phoneme_list = torch.zeros(T).type_as(phoneme_list)
    encoded_sil = label_encoder.encode_label('sil')
    torch.fill_(fl_phoneme_list, encoded_sil)

    # fill the output
    for phoneme, (start_time, end_time) in zip(phoneme_list, segmentation):
        # print(phoneme, start_time, end_time)
        start_index = int(start_time / duration * T)
        end_index = int(end_time / duration * T)
        fl_phoneme_list[start_index: end_index] = phoneme

    return fl_phoneme_list


def generate_boundary_seq(id, feat, duration, segmentation):
    """
    Generate boundary sequence.

    Parameters
    ----------
    feat : torch.Tensor
        (T, C), features
    duration : float
        total duration (in second)
    segmentation : list
        len = L, a list of tuples (start_second, end_second)

    Returns
    -------
    boundary_seq : torch.Tensor
        (T), binary tensor indicating if a frame is a start frame (=1) or not (=0)
    phn_end_seq : torch.Tensor
        (T), binary tensor indicating if a frame is an end frame (=1) or not (=0)

    """
    T = feat.shape[0]  # total number of frames

    boundary_seq = torch.zeros(T)
    boundary_seq[0] = 1
    for start_time, _ in segmentation[1:]:
        start_index = int(start_time / duration * T)
        # assert boundary_seq[start_index] == 0
        count = 0
        while boundary_seq[start_index] == 1:
            start_index += 1
            count += 1
            # print(f'move {count}')
        boundary_seq[start_index] = 1

    phn_end_seq = torch.zeros(len(segmentation))
    for i, (_, end_time) in enumerate(segmentation):
        # end_index = int(end_time / duration * T)
        end_index = int(end_time * 16000)
        phn_end_seq[i] = end_index
    return boundary_seq, phn_end_seq


def get_label_encoder(hparams):
    """
    Get label encoder.

    Parameters
    ----------
    hparams : dict
        Loaded hparams.

    Returns
    -------
    label_encoder : sb.dataio.encoder.CTCTextEncoder
        The label encoder for the dataset.
    """
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    phoneme_set = hparams['prepare']['phoneme_set_handler'].get_phoneme_set()
    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    label_encoder.insert_blank(index=hparams['blank_index'])
    return label_encoder


def prepare_datasets(hparams):
    logger.info('Preparing datasets.')
    computed_dataset_dir = Path(hparams['prepare']['dataset_dir']).parent / 'computed_dataset'

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
        datasets = data_io_prep(hparams)
        for set_name, dataset in zip(set_names, datasets):  # prepare dataset for train, valid, and test sets
            pkl_path = computed_dataset_dir / f'{set_name}.pkl'
            pkl_path.parent.mkdir(exist_ok=True)
            computed_dataset_dict = {}
            for data_sample in tqdm(dataset):  # for each data sample
                data_id = data_sample['id']  # get ID
                data_sample_dict = {}  # data sample content as a dictionary
                for key in output_keys:
                    if key != 'id':
                        data_sample_dict[key] = copy.deepcopy(data_sample[key]) # use deep copy to prevent errors
                computed_dataset_dict[data_id] = data_sample_dict
            # save the computed dataset
            with open(pkl_path, 'wb') as f:
                pickle.dump(computed_dataset_dict, f)
    else:
        logger.info('Load pre-computed datasets.')

    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        with open(pkl_path, 'rb') as f:
            computed_dataset_dict = pickle.load(f)
        computed_dataset = sb.dataio.dataset.DynamicItemDataset(computed_dataset_dict, output_keys=output_keys)
        computed_datasets.append(computed_dataset)

    label_encoder = get_label_encoder(hparams)

    # save label_encoder
    label_encoder_path = computed_dataset_dir / 'label_encoder.txt'
    label_encoder.save(label_encoder_path)

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

    label_encoder = get_label_encoder(hparams)

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
    @speechbrain.utils.data_pipeline.takes('flvl_gt_phn_seq', 'flvl_gt_cnncl_seq', 'aug_flvl_gt_phn_seq', 'aug_flvl_gt_cnncl_seq')
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
    prior = torch.zeros(len(label_encoder))
    for train_sample in tqdm(train_dataset):
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