import logging
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import copy
import librosa

import torch
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline

from utils.data_io import output_keys, get_label_encoder, generate_boundary_seq

logger = logging.getLogger(__name__)
def prepare_datasets(hparams):
    logger.info('Preparing datasets.')
    dataset_dir = Path(hparams['prepare']['dataset_dir']).parent

    # prepare dataset or load pre-computed datasets
    computed_datasets = []

    set_names = ['train', 'valid', 'test']
    computed_dataset_dir = dataset_dir / 'computed_dataset'
    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        # check if pre-computed datasets exist
        if not pkl_path.exists():
            raise FileNotFoundError(f'pre-computed dataset not found: {pkl_path.absolute()}')
        with open(pkl_path, 'rb') as f:
            computed_dataset_dict = pickle.load(f)
        computed_dataset = sb.dataio.dataset.DynamicItemDataset(computed_dataset_dict, output_keys=output_keys)
        computed_datasets.append(computed_dataset)

    # get label encoder
    label_encoder = get_label_encoder(hparams)

    # get external data
    dnn_hmm_results_path = dataset_dir / 'external_data' / 'dnn_hmm_test.json'
    if dnn_hmm_results_path.exists():
        with open(dnn_hmm_results_path) as f:
            dnn_hmm_results = json.load(f)

        @speechbrain.utils.data_pipeline.takes('id')
        @speechbrain.utils.data_pipeline.provides('ext_dnn_hmm_seg_seq', 'ext_dnn_hmm_phn_seq')
        def dnn_hmm_pipeline(utt_id):
            dnn_hmm_result = dnn_hmm_results[utt_id]

            seg_seq = []
            phn_seq = []
            for start_time, end_time, phn in dnn_hmm_result:
                seg_seq.append([start_time, end_time])
                if '*' in phn:
                    phn = 'sil'
                phn_seq.append(label_encoder.encode_label(phn))

            yield torch.tensor(seg_seq)
            yield torch.tensor(phn_seq)
        sb.dataio.dataset.add_dynamic_item([computed_datasets[2]], dnn_hmm_pipeline)

        # boundary sequence
        @speechbrain.utils.data_pipeline.takes('id', 'feat', 'duration', 'ext_dnn_hmm_seg_seq')
        @speechbrain.utils.data_pipeline.provides('ext_dnn_hmm_boundary_seq', 'ext_dnn_hmm_phn_end_seq')
        def ext_dnn_hmm_boundary_seq_pipeline(id, feat, duration, ext_dnn_hmm_segmentation):
            boundary_seq, phn_end_seq = generate_boundary_seq(id, feat, duration, ext_dnn_hmm_segmentation)
            yield boundary_seq
            yield phn_end_seq

        sb.dataio.dataset.add_dynamic_item([computed_datasets[2]], ext_dnn_hmm_boundary_seq_pipeline)

        # MD label
        @speechbrain.utils.data_pipeline.takes('id', 'ext_dnn_hmm_phn_seq', 'gt_cnncl_seq')
        @speechbrain.utils.data_pipeline.provides('ext_plvl_dnn_hmm_md_lbl_seq')
        def plvl_gt_md_lbl_seq_pipeline(id, ext_dnn_hmm_phn_seq, gt_cnncl_seq):
            return torch.ne(ext_dnn_hmm_phn_seq, gt_cnncl_seq).long()

        sb.dataio.dataset.add_dynamic_item([computed_datasets[2]], plvl_gt_md_lbl_seq_pipeline)

        new_output_keys = output_keys + [
            'ext_dnn_hmm_seg_seq', 'ext_dnn_hmm_phn_seq',
            'ext_dnn_hmm_boundary_seq', 'ext_dnn_hmm_phn_end_seq',
            'ext_plvl_dnn_hmm_md_lbl_seq'
        ]
        sb.dataio.dataset.set_output_keys([computed_datasets[2]], new_output_keys)


    return computed_datasets, label_encoder

