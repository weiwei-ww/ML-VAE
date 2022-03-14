import time
from joblib import Parallel, delayed
import numpy as np

import torch


def log(x):
    eps = 1e-5

    ret = x.detach().cpu()
    ret[torch.logical_and(ret >= 0, ret < eps)] = eps

    return torch.log(ret).numpy()


def decode_boundary(
        eval_outputs,  # shape = (B, T, N)
        utt_ids,  # len = B
        lens,  # shape = (B)
        can_seqs,  # shape = (B, L, N)
        can_seq_lens,  # shape = (B)
        prior,  # shape = (N)
        **kwargs
):
    '''
    Decode the boundaries based on p(y|x), p(y), and p(b|x).

    Parameters
    ----------
    eval_outputs
    utt_ids
    lens
    can_seqs
    can_seq_lens
    prior
    kwargs

    Returns
    -------

    '''

    # pre-computation
    p_yx = torch.sigmoid(eval_outputs['phoneme_ret']).cpu()  # shape = (B, T, N)
    log_p_yx = log(p_yx)

    y = torch.argmax(can_seqs, dim=-1).cpu()  # shape = (B, L)

    boundary_beta = eval_outputs['boundary_beta'].cpu()  # shape = (B, T)
    p_b = torch.stack([boundary_beta, 1 - boundary_beta], dim=2)  # shape = (B, T, 2)
    log_p_b = log(p_b)  # shape = (B, T, 2)

    log_p_y = log(prior.cpu())

    decoded_boundaries = {}
    for i, utt_id in enumerate(utt_ids):
        T_i = lens[i]
        L_i = can_seq_lens[i]

        log_p_yx_i = log_p_yx[i, :T_i]  # shape = (T_i, N)
        y_i = y[i, :L_i]  # shape = (L_i)
        log_p_b_i = log_p_b[i, :T_i]  # shape = (T_i)

        dp_value = np.zeros((L_i, T_i), dtype=float)
        dp_value.fill(-np.inf)

        dp_path = np.zeros((L_i, T_i), dtype=int)
        dp_path.fill(-1)

        dp_value[:, 0] = -np.inf
        dp_value[0, 0] = log_p_yx_i[0, y_i[0]] - log_p_y[y_i[0]]

        # decode with dynamic programming
        for l in range(L_i):
            for t in range(1, T_i):
                if l == 0:
                    dp_value[l, t] = dp_value[l, t - 1] + log_p_b_i[t, 0] + \
                                     log_p_yx_i[t, y_i[l]] - log_p_y[y_i[0]]
                    dp_path[l, t] = 0
                else:
                    left_value = dp_value[l, t - 1] + log_p_b_i[t, 0] + \
                                 log_p_yx_i[t, y_i[l]] - log_p_y[y_i[l]]  # b = 0
                    up_value = dp_value[l - 1, t - 1] + log_p_b_i[t, 1] + \
                               log_p_yx_i[t, y_i[l]] - log_p_y[y_i[l]]  # b = 1
                    if left_value > up_value:
                        dp_value[l, t] = left_value
                        dp_path[l, t] = 0
                    else:
                        dp_value[l, t] = up_value
                        dp_path[l, t] = 1

        # backtracking
        l, t = L_i - 1, T_i - 1
        boundary_index = []
        while t > 0:
            if dp_path[l, t] == 1:
                l -= 1
                boundary_index.append(int(t))
            t -= 1
        boundary_index.append(int(t))
        assert l == t == 0
        boundary_index.reverse()

        decoded_boundary_seq = np.zeros(T_i, dtype=int)
        decoded_boundary_seq[boundary_index] = 1
        assert decoded_boundary_seq.sum() == L_i

        decoded_boundaries[utt_id] = np.array(decoded_boundary_seq)

    return decoded_boundaries


def decode_phoneme_mdd_conditional(
        eval_outputs,  # shape = (B, T, N)
        utt_ids,  # len = B
        lens,  # shape = (B)
        can_seqs,  # shape = (B, L, N)
        can_seq_lens,  # shape = (B)
        prior,  # shape = (N)
        **kwargs
):
    '''
    Decode the beta sequence based on the decoded boundaries.

    Parameters
    ----------
    eval_outputs
    utt_ids
    lens
    can_seqs
    can_seq_lens
    prior
    kwargs

    Returns
    -------

    '''

    # pre-computation
    phoneme_ret = torch.sigmoid(eval_outputs['phoneme_ret']).cpu()  # shape = (B, T, N)
    p_yx = torch.stack([phoneme_ret, 1 - phoneme_ret], dim=3)  # shape = (B, T, N, 2)
    log_p_yx = log(p_yx)  # shape = (B, T, N, 2)

    py = torch.stack([prior, 1 - prior], dim=1).cpu()  # shape = (N, 2)
    log_py = log(py)

    y = torch.argmax(can_seqs, dim=-1).cpu()  # shape = (B, L)

    p_beta = eval_outputs['vae_mdd_soft_beta'].cpu()  # shape = (B, T, 2)
    log_p_beta = log(p_beta)

    decoded_boundaries = eval_outputs['decoded_boundaries']

    phoneme_mdd_ret = torch.zeros_like(y)  # shape = (B, L)
    for i, utt_id in enumerate(utt_ids):
        T_i = lens[i]
        L_i = can_seq_lens[i]

        boundaries_i = decoded_boundaries[utt_id]  # shape = (L)
        boundaries_i = np.where(boundaries_i == 1)[0]

        for j in range(len(boundaries_i)):
            start_index = boundaries_i[j]
            if j < len(boundaries_i) - 1:
                end_index = boundaries_i[j + 1]
            else:
                end_index = T_i

            y_ij = y[i, j]

            # log prob for correct pronunciation
            log_p_correct = 0
            # log_p_correct += np.sum(log_p_beta[i, start_index: end_index, 0])
            log_p_correct += np.sum(log_p_yx[i, start_index: end_index, y_ij, 0])
            log_p_correct -= log_py[y_ij, 0] * (end_index - start_index)

            # log prob for mispronunciation
            log_p_mispronunciation = 0
            # log_p_mispronunciation += np.sum(log_p_beta[i, start_index: end_index, 1])
            log_p_mispronunciation += np.sum(log_p_yx[i, start_index: end_index, y_ij, 1])
            log_p_mispronunciation -= log_py[y_ij, 1] * (end_index - start_index)

            if log_p_mispronunciation > log_p_correct:
                phoneme_mdd_ret[i, j] = 1

    return phoneme_mdd_ret


def decode_plvl_md_lbl_seqs_full_non_par(
        predictions,
        utt_ids,  # len = B
        feat_lens,  # shape = (B)
        plvl_cnnl_seqs,  # shape = (B, L, N)
        plvl_cnnl_seq_lens,  # shape = (B)
        prior,  # shape = (N)
        weight=1.0
):
    '''
    Decode the boundaries and the beta sequence simultaneously. Non-parallel version.

    Parameters
    ----------
    predictions : dict
        Model outputs.
    utt_ids ： list
        A list of IDs.
    feat_lens : torch.Tensor
        Frame lengths of each sample in the batch (relative length).
    plvl_cnnl_seqs : torch.Tensor
        Phoneme level canonical phoneme sequences.
    plvl_cnnl_seq_lens : torch.Tensor
        Phoneme lengths of each sample in the batch (relative length).
    prior : torch.Tensor
        Prior distribution of the phonemes.
    weight : float
        Weight for decoding.

    Returns
    -------

    '''
    # get the absolute lengths
    feat_lens = torch.round(feat_lens * predictions['phn_recog_out'].shape[1]).int()
    plvl_cnnl_seq_lens = torch.round(plvl_cnnl_seq_lens * plvl_cnnl_seqs.shape[1]).int()

    # pre-computation
    phn_recog_out = torch.sigmoid(predictions['phn_recog_out'])  # shape = (B, T, N)
    p_yx = torch.stack([phn_recog_out, 1 - phn_recog_out], dim=3)  # shape = (B, T, N, 2)
    log_p_yx = log(p_yx)  # shape = (B, T, N, 2)

    p_y = torch.stack([prior, 1 - prior], dim=1)  # shape = (N, 2)
    log_p_y = log(p_y)  # shape = (N, 2)

    y = plvl_cnnl_seqs  # shape = (B, L)

    boundary_v = predictions['boundary_v']  # shape = (B, T)
    p_b = torch.stack([boundary_v, 1 - boundary_v], dim=2)  # shape = (B, T, 2)
    log_p_b = log(p_b)  # shape = (B, T, 2)

    pi_logits = predictions['pi_logits']  # shape = (B, T, 2)
    p_pi = torch.softmax(pi_logits, dim=-1)  # shape = (B, T, 2)
    log_p_pi = log(p_pi)  # shape = (B, T, 2)

    # decoded_boundary_seqs = {}
    # flvl_md_lbl_seqs = torch.zeros_like(boundary_v)  # shape = (B, T)
    # plvl_md_lbl_seqs = torch.zeros_like(y)  # shape = (B, L)
    
    decoded_boundary_seqs = []
    flvl_md_lbl_seqs = []
    plvl_md_lbl_seqs = []

    start_time = time.time()
    for i, utt_id in enumerate(utt_ids):
        T_i = feat_lens[i]
        L_i = plvl_cnnl_seq_lens[i]

        log_p_yx_i = log_p_yx[i, :T_i, :]  # shape = (T_i, N, 2)
        y_i = y[i, :L_i]  # shape = (L_i)
        log_p_b_i = log_p_b[i, :T_i, :]  # shape = (T_i, 2)
        log_p_pi_i = log_p_pi[i, :T_i, :]  # shape = (T_i, 2)

        dp_value = np.zeros((L_i, T_i, 2), dtype=float)
        dp_value.fill(-np.inf)

        dp_path = np.zeros((L_i, T_i, 2), dtype=int)
        dp_path.fill(-1)

        dp_value[0, 0, 0] = weight * log_p_pi_i[0, 0] + log_p_yx_i[0, y_i[0], 0] - log_p_y[y_i[0], 0]
        dp_value[0, 0, 1] = weight * log_p_pi_i[0, 1] + log_p_yx_i[0, y_i[0], 1] - log_p_y[y_i[0], 1]

        # decode with dynamic programming
        for l in range(L_i):
            for t in range(1, T_i):
                if l == 0:
                    # correct pronunciation
                    dp_value[l, t, 0] = \
                        dp_value[l, t - 1, 0] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]
                    dp_path[l, t, 0] = 0
                    # mispronunciation
                    dp_value[l, t, 1] = \
                        dp_value[l, t - 1, 1] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]
                    dp_path[l, t, 1] = 0
                else:
                    # correct pronunciation
                    hold_value = \
                        dp_value[l, t - 1, 0] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 0
                    from_correct_value = \
                        dp_value[l - 1, t - 1, 0] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 1
                    from_incorrect_value = \
                        dp_value[l - 1, t - 1, 1] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 1

                    value_list = [hold_value, from_correct_value, from_incorrect_value]
                    dp_value[l, t, 0] = np.max(value_list)
                    dp_path[l, t, 0] = np.argmax(value_list)

                    # mispronunciation
                    hold_value = \
                        dp_value[l, t - 1, 1] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 0
                    from_correct_value = \
                        dp_value[l - 1, t - 1, 0] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 1] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 1
                    from_incorrect_value = \
                        dp_value[l - 1, t - 1, 1] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 1] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 1

                    value_list = [hold_value, from_correct_value, from_incorrect_value]
                    dp_value[l, t, 1] = np.max(value_list)
                    dp_path[l, t, 1] = np.argmax(value_list)

        # backtracking
        l, t = L_i - 1, T_i - 1
        boundary_index_i = []
        frame_mdd_ret_i = []
        phoneme_mdd_ret_i = []

        if dp_value[l, t, 0] > dp_value[l, t, 1]:
            frame_mdd_ret_i.append(0)
            phoneme_mdd_ret_i.append(0)
            beta = 0
        else:
            phoneme_mdd_ret_i.append(1)
            frame_mdd_ret_i.append(1)
            beta = 1
        while t > 0:
            if dp_path[l, t, beta] == 1:
                l -= 1
                boundary_index_i.append(int(t))
                frame_mdd_ret_i.append(0)
                phoneme_mdd_ret_i.append(0)
                beta = 0
            elif dp_path[l, t, beta] == 2:
                l -= 1
                boundary_index_i.append(int(t))
                frame_mdd_ret_i.append(1)
                phoneme_mdd_ret_i.append(1)
                beta = 1
            else:
                frame_mdd_ret_i.append(frame_mdd_ret_i[-1])
            t -= 1
        boundary_index_i.append(int(t))
        assert l == t == 0, 'l = {}, t = {}'.format(l, t)
        boundary_index_i.reverse()
        frame_mdd_ret_i.reverse()
        phoneme_mdd_ret_i.reverse()

        # convert boundary indices into binary boundaries
        decoded_boundary_seq = np.zeros(T_i, dtype=int)
        decoded_boundary_seq[boundary_index_i] = 1
        assert decoded_boundary_seq.sum() == L_i
        
        # save decoded boundaries
        decoded_boundary_seqs.append(decoded_boundary_seq)

        # save phoneme mdd results
        flvl_md_lbl_seqs.append(frame_mdd_ret_i)
        plvl_md_lbl_seqs.append(phoneme_mdd_ret_i)
    end_time = time.time()
    print(f'Time elapsed for decoding: {round(end_time - start_time, 2)} seconds')

    # print(f'{np.sum([np.sum(l) for l in flvl_md_lbl_seqs])}/{np.sum([len(l) for l in flvl_md_lbl_seqs])}')
    # print(f'{np.sum([np.sum(l) for l in plvl_md_lbl_seqs])}/{np.sum([len(l) for l in plvl_md_lbl_seqs])}')

    return decoded_boundary_seqs, flvl_md_lbl_seqs, plvl_md_lbl_seqs


def decode_plvl_md_lbl_seqs_full(
        predictions,
        utt_ids,  # len = B
        feat_lens,  # shape = (B)
        plvl_cnnl_seqs,  # shape = (B, L, N)
        plvl_cnnl_seq_lens,  # shape = (B)
        prior,  # shape = (N)
        weight=1.0
):
    '''
    Decode the boundaries and the pi sequence simultaneously. The parallel version.

    Parameters
    ----------
    predictions : dict
        Model outputs.
    utt_ids ： list
        A list of IDs.
    feat_lens : torch.Tensor
        Frame lengths of each sample in the batch (relative length).
    plvl_cnnl_seqs : torch.Tensor
        Phoneme level canonical phoneme sequences.
    plvl_cnnl_seq_lens : torch.Tensor
        Phoneme lengths of each sample in the batch (relative length).
    prior : torch.Tensor
        Prior distribution of the phonemes.
    weight : float
        Weight for decoding.

    Returns
    -------

    '''
    # get the absolute lengths
    feat_lens = torch.round(feat_lens * predictions['phn_recog_out'].shape[1]).int()
    plvl_cnnl_seq_lens = torch.round(plvl_cnnl_seq_lens * plvl_cnnl_seqs.shape[1]).int()

    # pre-computation
    phn_recog_out = torch.sigmoid(predictions['phn_recog_out'])  # shape = (B, T, N)
    p_yx = torch.stack([phn_recog_out, 1 - phn_recog_out], dim=3)  # shape = (B, T, N, 2)
    log_p_yx = log(p_yx)  # shape = (B, T, N, 2)

    p_y = torch.stack([prior, 1 - prior], dim=1)  # shape = (N, 2)
    log_p_y = log(p_y)  # shape = (N, 2)

    y = plvl_cnnl_seqs  # shape = (B, L)

    boundary_v = predictions['boundary_v']  # shape = (B, T)
    p_b = torch.stack([boundary_v, 1 - boundary_v], dim=2)  # shape = (B, T, 2)
    log_p_b = log(p_b)  # shape = (B, T, 2)

    pi_logits = predictions['pi_logits']  # shape = (B, T, 2)
    p_pi = torch.softmax(pi_logits, dim=-1)  # shape = (B, T, 2)
    log_p_pi = log(p_pi)  # shape = (B, T, 2)

    decoded_boundary_seqs = []
    flvl_md_lbl_seqs = []
    plvl_md_lbl_seqs = []

    # start_time = time.time()

    # convert to np.array
    feat_lens = feat_lens.cpu().numpy()
    plvl_cnnl_seq_lens = plvl_cnnl_seq_lens.cpu().numpy()
    y = y.cpu().numpy()

    def decode_one_utt(i):
        T_i = feat_lens[i]
        L_i = plvl_cnnl_seq_lens[i]

        log_p_yx_i = log_p_yx[i, :T_i, :]  # shape = (T_i, N, 2)
        y_i = y[i, :L_i]  # shape = (L_i)
        log_p_b_i = log_p_b[i, :T_i, :]  # shape = (T_i, 2)
        log_p_pi_i = log_p_pi[i, :T_i, :]  # shape = (T_i, 2)

        dp_value = np.zeros((L_i, T_i, 2), dtype=float)
        dp_value.fill(-np.inf)

        dp_path = np.zeros((L_i, T_i, 2), dtype=int)
        dp_path.fill(-1)

        dp_value[0, 0, 0] = weight * log_p_pi_i[0, 0] + log_p_yx_i[0, y_i[0], 0] - log_p_y[y_i[0], 0]
        dp_value[0, 0, 1] = weight * log_p_pi_i[0, 1] + log_p_yx_i[0, y_i[0], 1] - log_p_y[y_i[0], 1]

        # decode with dynamic programming
        for l in range(L_i):
            for t in range(1, T_i):
                if l == 0:
                    # correct pronunciation
                    dp_value[l, t, 0] = \
                        dp_value[l, t - 1, 0] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]
                    dp_path[l, t, 0] = 0
                    # mispronunciation
                    dp_value[l, t, 1] = \
                        dp_value[l, t - 1, 1] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]
                    dp_path[l, t, 1] = 0
                else:
                    # correct pronunciation
                    hold_value = \
                        dp_value[l, t - 1, 0] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 0
                    from_correct_value = \
                        dp_value[l - 1, t - 1, 0] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 1
                    from_incorrect_value = \
                        dp_value[l - 1, t - 1, 1] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 0] - log_p_y[y_i[l], 0]  # b = 1

                    value_list = [hold_value, from_correct_value, from_incorrect_value]
                    dp_value[l, t, 0] = np.max(value_list)
                    dp_path[l, t, 0] = np.argmax(value_list)

                    # mispronunciation
                    hold_value = \
                        dp_value[l, t - 1, 1] + log_p_b_i[t, 0] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 0
                    from_correct_value = \
                        dp_value[l - 1, t - 1, 0] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 1] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 1
                    from_incorrect_value = \
                        dp_value[l - 1, t - 1, 1] + log_p_b_i[t, 1] + weight * log_p_pi_i[t, 1] + \
                        log_p_yx_i[t, y_i[l], 1] - log_p_y[y_i[l], 1]  # b = 1

                    value_list = [hold_value, from_correct_value, from_incorrect_value]
                    dp_value[l, t, 1] = np.max(value_list)
                    dp_path[l, t, 1] = np.argmax(value_list)

        # backtracking
        l, t = L_i - 1, T_i - 1
        boundary_index_i = []
        frame_mdd_ret_i = []
        phoneme_mdd_ret_i = []

        if dp_value[l, t, 0] > dp_value[l, t, 1]:
            frame_mdd_ret_i.append(0)
            phoneme_mdd_ret_i.append(0)
            beta = 0
        else:
            phoneme_mdd_ret_i.append(1)
            frame_mdd_ret_i.append(1)
            beta = 1
        while t > 0:
            if dp_path[l, t, beta] == 1:
                l -= 1
                boundary_index_i.append(int(t))
                frame_mdd_ret_i.append(0)
                phoneme_mdd_ret_i.append(0)
                beta = 0
            elif dp_path[l, t, beta] == 2:
                l -= 1
                boundary_index_i.append(int(t))
                frame_mdd_ret_i.append(1)
                phoneme_mdd_ret_i.append(1)
                beta = 1
            else:
                frame_mdd_ret_i.append(frame_mdd_ret_i[-1])
            t -= 1
        boundary_index_i.append(int(t))
        assert l == t == 0, 'l = {}, t = {}'.format(l, t)
        boundary_index_i.reverse()
        frame_mdd_ret_i.reverse()
        phoneme_mdd_ret_i.reverse()

        # convert boundary indices into binary boundaries
        decoded_boundary_seq = np.zeros(T_i, dtype=int)
        decoded_boundary_seq[boundary_index_i] = 1
        assert decoded_boundary_seq.sum() == L_i

        return decoded_boundary_seq, frame_mdd_ret_i, phoneme_mdd_ret_i

    B = len(utt_ids)
    decoded_result_list = Parallel(n_jobs=B)(delayed(decode_one_utt)(i) for i in range(B))
    for decoded_boundary_seq, frame_mdd_ret_i, phoneme_mdd_ret_i in decoded_result_list:
        decoded_boundary_seqs.append(decoded_boundary_seq)
        flvl_md_lbl_seqs.append(frame_mdd_ret_i)
        plvl_md_lbl_seqs.append(phoneme_mdd_ret_i)

    # for i in range(len(utt_ids)):
    #     decoded_boundary_seq, frame_mdd_ret_i, phoneme_mdd_ret_i = decode_one_utt(i)
    #     decoded_boundary_seqs.append(decoded_boundary_seq)
    #     flvl_md_lbl_seqs.append(frame_mdd_ret_i)
    #     plvl_md_lbl_seqs.append(phoneme_mdd_ret_i)

    # end_time = time.time()
    # print(f'Time elapsed for decoding: {round(end_time - start_time, 2)} seconds')

    # print(f'{np.sum([np.sum(l) for l in flvl_md_lbl_seqs])}/{np.sum([len(l) for l in flvl_md_lbl_seqs])}')
    # print(f'{np.sum([np.sum(l) for l in plvl_md_lbl_seqs])}/{np.sum([len(l) for l in plvl_md_lbl_seqs])}')

    return decoded_boundary_seqs, flvl_md_lbl_seqs, plvl_md_lbl_seqs
