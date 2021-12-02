import torch


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
        boundary_seq[start_index] = 1

    phn_end_seq = torch.zeros(len(segmentation))
    for i, (_, end_time) in enumerate(segmentation):
        # end_index = int(end_time / duration * T)
        end_index = int(end_time * 16000)
        phn_end_seq[i] = end_index
    return boundary_seq, phn_end_seq
