import json
import logging
from tqdm import tqdm
import pickle
from pathlib import Path
import subprocess

import torch
import speechbrain as sb
import speechbrain.dataio.encoder

logger = logging.getLogger(__name__)


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


def compute_fbank_kaldi(wav_scp_path, feature_params):
    """
    Compute the fbank feature with Kaldi.

    Parameters
    ----------
    wav_scp_path : Path
        Path to the wav.scp file.

    feature_params : dict
        Parameters for feature extraction.

    Returns
    -------
    None
    """

    def convert_utt2spk_to_spk2utt(utt2spk_path):
        # load utt2spk
        with open(utt2spk_path) as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]

        # analyze data
        data = {}
        for utt_id, spk_id in lines:
            if spk_id not in data:
                data[spk_id] = []
            data[spk_id].append(utt_id)

        # create spk2utt
        lines = []
        for spk_id, utt_ids in data.items():
            line = ' '.join([spk_id] + utt_ids) + '\n'
            lines.append(line)

        return lines

    def run_cmd(cmd):
        cmd = ' '.join(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        if process.returncode != 0:
            raise ValueError(f'non-zero return code: {process.returncode}')

    set_name = wav_scp_path.stem.split('.')[0]
    assert set_name in ['train', 'valid', 'test']
    kaldi_data_dir = wav_scp_path.parent

    # parameters
    sr = feature_params['sample_rate']
    hop_length = feature_params['hop_length']
    n_fft = feature_params['n_fft']
    n_mels = feature_params['n_mels']

    # compute fbank with Kaldi
    cmd = ['compute-fbank-feats --window-type=hamming --htk-compat=true --dither=0.0 --energy-floor=1.0 --snip-edges=false']

    # feature extraction parameters
    cmd.append(f'--sample-frequency={sr}')
    cmd.append(f'--frame-shift={hop_length}')
    cmd.append(f'--frame-length={n_fft / sr * 1000}')
    cmd.append(f'--num-mel-bins={n_mels}')

    # IO ark/scp files
    cmd.append(f'scp:{wav_scp_path.absolute()}')
    raw_feats_scp_path = kaldi_data_dir / f'{set_name}.raw_feats.scp'
    raw_feats_ark_path = raw_feats_scp_path.with_suffix('.ark')
    cmd.append(f'ark,scp:{raw_feats_ark_path.absolute()},{raw_feats_scp_path.absolute()}')

    run_cmd(cmd)

    # add deltas
    cmd = ['add-deltas']
    cmd.append(f'scp:{raw_feats_scp_path.absolute()}')
    delta_feats_scp_path = kaldi_data_dir / f'{set_name}.delta_feats.scp'
    delta_feats_ark_path = delta_feats_scp_path.with_suffix('.ark')
    cmd.append(f'ark,scp:{delta_feats_ark_path.absolute()},{delta_feats_scp_path.absolute()}')

    run_cmd(cmd)

    # compute CMVN
    utt2spk_path = kaldi_data_dir / f'{set_name}.utt2spk'
    spk2utt_lines = convert_utt2spk_to_spk2utt(utt2spk_path)
    spk2utt_content = ''.join(spk2utt_lines)
    cmd = [f'echo "{spk2utt_content}" | compute-cmvn-stats --spk2utt=ark:-']
    cmd.append(f'scp:{delta_feats_scp_path.absolute()}')
    cmvn_scp_path = kaldi_data_dir / f'{set_name}.cmvn.scp'
    cmvn_ark_path = cmvn_scp_path.with_suffix('.ark')
    cmd.append(f'ark,scp:{cmvn_ark_path.absolute()},{cmvn_scp_path.absolute()}')

    run_cmd(cmd)

    # apply CMVN
    cmd = ['apply-cmvn --norm-vars=true']
    cmd.append(f'--utt2spk=ark:{utt2spk_path.absolute()}')
    cmd.append(f'scp:{cmvn_scp_path.absolute()}')
    cmd.append(f'scp:{delta_feats_scp_path.absolute()}')
    feats_scp_path = kaldi_data_dir / f'{set_name}.feats.scp'
    feats_ark_path = feats_scp_path.with_suffix('.ark')
    cmd.append(f'ark,scp:{feats_ark_path.absolute()},{feats_scp_path.absolute()}')

    run_cmd(cmd)


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


def apply_flvl_saved_md_results(x, saved_md_results):
    """
    Apply saved MD results to x.

    Parameters
    ----------
    x : torch.Tensor
        Data tensor.

    saved_md_results : list
        A list of saved MD results.

    Returns
    -------
    updated_x: torch.Tensor
        Updated x.
    """
    invalid_indices = []
    for _, start_pct, end_pct in saved_md_results:
        start_index = round(start_pct * len(x))
        end_index = round(end_pct * len(x))
        invalid_indices += list(range(start_index, end_index))
    valid_indices = [i for i in range(len(x)) if i not in invalid_indices]
    if isinstance(x, list):
        return [x[i] for i in valid_indices]
    else:
        return x[valid_indices]


def apply_plvl_saved_md_results(x, saved_md_results):
    """
    Apply saved MD results to x.

    Parameters
    ----------
    x : torch.Tensor
        Data tensor.

    saved_md_results : list
        A list of saved MD results.

    Returns
    -------
    updated_x: torch.Tensor
        Updated x.
    """
    invalid_indices = [idx for idx, _, _ in saved_md_results]
    valid_indices = [i for i in range(len(x)) if i not in invalid_indices]
    if isinstance(x, list):
        return [x[i] for i in valid_indices]
    else:
        return x[valid_indices]


def apply_boundary_saved_md_results(x, saved_md_results):
    """
    Apply saved MD results to x.

    Parameters
    ----------
    x : torch.Tensor
        Data tensor.

    saved_md_results : list
        A list of saved MD results.

    Returns
    -------
    updated_x: torch.Tensor
        Updated x.
    """
    idx_seq = torch.where(x == 1)[0].tolist()
    idx_seq.append(len(x))

    invalid_indices = []
    for idx, _, _ in saved_md_results:
        start_idx = idx_seq[idx]
        end_idx = idx_seq[idx + 1]
        invalid_indices += list(range(start_idx, end_idx))

    valid_indices = [i for i in range(len(x)) if i not in invalid_indices]
    if isinstance(x, list):
        return [x[i] for i in valid_indices]
    else:
        return x[valid_indices]
