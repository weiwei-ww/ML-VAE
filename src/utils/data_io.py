import librosa

import torch
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline

from utils.preprocessing import generate_flvl_annotation


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
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_segmentation', 'phoneme_seq')
    @speechbrain.utils.data_pipeline.provides('gt_phn', 'flvl_gt_phn', 'aug_flvl_gt_phn')
    def flvl_phoneme_pipeline(feat, aug_feat, duration, segmentation, phoneme_seq):
        gt_phn = label_encoder.encode_sequence_torch(phoneme_seq)
        yield gt_phn  # encoded phonemes
        flvl_gt_phn = generate_flvl_annotation(label_encoder, feat, duration, segmentation, gt_phn)
        yield flvl_gt_phn  # frame level phonemes
        aug_flvl_gt_phn = generate_flvl_annotation(label_encoder, aug_feat, duration, segmentation, gt_phn)
        yield aug_flvl_gt_phn  # frame level phonemes with augmentation
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_phoneme_pipeline)

    # frame level canonicals
    @speechbrain.utils.data_pipeline.takes('feat', 'aug_feat', 'duration', 'gt_segmentation', 'gt_cnncl')
    @speechbrain.utils.data_pipeline.provides('gt_cnncl', 'flvl_gt_cnncl', 'aug_flvl_gt_cnncl')
    def flvl_canonical_pipeline(feat, aug_feat, duration, segmentation, gt_cnncl):
        gt_cnncl = label_encoder.encode_sequence_torch(gt_cnncl)
        yield gt_cnncl  # encoded canonicals
        flvl_gt_cnncl = generate_flvl_annotation(label_encoder, feat, duration, segmentation, gt_cnncl)
        yield flvl_gt_cnncl  # frame level canonicals
        aug_flvl_gt_cnncl = generate_flvl_annotation(label_encoder, aug_feat, duration, segmentation, gt_cnncl)
        yield aug_flvl_gt_cnncl  # frame level canonicals with augmentation
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_canonical_pipeline)

    # MD ground truth pipelines
    # phoneme level ground truth
    @speechbrain.utils.data_pipeline.takes('gt_phn', 'gt_cnncl')
    @speechbrain.utils.data_pipeline.provides('plvl_gt_md_lbl')
    def plvl_gt_md_lbl_pipeline(gt_phn, gt_cnncl):
        return torch.ne(gt_phn, gt_cnncl).long()
    sb.dataio.dataset.add_dynamic_item(datasets, plvl_gt_md_lbl_pipeline)

    # frame level ground truth
    @speechbrain.utils.data_pipeline.takes('flvl_gt_phn', 'flvl_gt_cnncl', 'aug_flvl_gt_phn', 'aug_flvl_gt_cnncl')
    @speechbrain.utils.data_pipeline.provides('flvl_gt_md_lbl', 'aug_flvl_gt_md_lbl')
    def flvl_gt_md_lbl_pipeline(flvl_gt_phn, flvl_gt_cnncl, aug_flvl_gt_phn, aug_flvl_gt_cnncl):
        flvl_gt_md_lbl = torch.ne(flvl_gt_phn, flvl_gt_cnncl).long()
        yield flvl_gt_md_lbl
        aug_flvl_gt_md_lbl = torch.ne(aug_flvl_gt_phn, aug_flvl_gt_cnncl).long()
        yield aug_flvl_gt_md_lbl
    sb.dataio.dataset.add_dynamic_item(datasets, flvl_gt_md_lbl_pipeline)

    # set output keys
    output_keys = [
        'id',
        'wav', 'aug_wav',  # wave form
        'feat', 'aug_feat',  # feature
        'gt_phn', 'gt_cnncl',  # encoded phonemes
        'flvl_gt_phn', 'flvl_gt_cnncl',  # frame level phonemes
        'aug_flvl_gt_cnncl', 'aug_flvl_gt_cnncl',  # frame level phoneme with augmentation
        'plvl_gt_md_lbl', 'flvl_gt_md_lbl', 'aug_flvl_gt_md_lbl',  # phoneme and frame level MD ground truth
        'gt_segmentation'  # ground truth segmentation
    ]
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    # fit the label encoder
    phoneme_set = hparams['prepare']['phoneme_set_handler'].get_phoneme_set()
    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    label_encoder.insert_blank(index=hparams['blank_index'])

    print(datasets[0][0])

    return train_dataset, valid_dataset, test_dataset, label_encoder