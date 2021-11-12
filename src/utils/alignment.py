import torch
import speechbrain as sb
import speechbrain.utils.edit_distance
import speechbrain.dataio.wer

# from speechbrain.utils.edit_distance import op_table
# from speechbrain.utils.edit_distance import alignment
# from speechbrain.utils.edit_distance import count_ops


def align_sequences(a, b, c=None, empty_value=-1):
    """
    Align two or three sequences.

    Parameters
    ----------
    a : torch.Tensor or list
    b : torch.Tensor of list
    c : torch.Tensor of list
        sequences to be aligned
    empty_value : any
        value to use for insertion and deletion

    Returns
    -------
    ali_a : list
    ali_b : list
    ali_c : list
        aligned sequences, with same length
    """
    # check input
    def convert_to_list(x):
        if isinstance(x, torch.Tensor):
            if x.ndim > 1:
                raise ValueError('Only one-dimension input is allowed')
            x = x.tolist()
        elif not isinstance(x, list):
            raise TypeError(f'Unsupported input type {type(x).__name__}')
        return x

    a = convert_to_list(a)
    b = convert_to_list(b)
    if c is not None:
        c = convert_to_list(c)

    # perform alignment
    op_table = sb.utils.edit_distance.op_table(a, b)
    alignment = sb.utils.edit_distance.alignment(op_table)

    # get aligned sequences
    ali_a, ali_b, ali_c = [], [], []
    for _, a_index, b_index in alignment:
        ali_a.append(a[a_index] if a_index is not None else empty_value)
        ali_b.append(b[b_index] if b_index is not None else empty_value)
        if c is not None:
            ali_c.append(b[b_index] if b_index is not None else empty_value)

    if c is not None:
        assert len(ali_a) == len(ali_b) == len(ali_c)
        return ali_a, ali_b, ali_c
    else:
        assert len(ali_a) == len(ali_b)
        return ali_a, ali_b


def batch_align_sequences(batch_a, batch_b, batch_c=None):
    """
    Preform alignment on batches.

    Parameters
    ----------
    batch_a : list
    batch_b : list
    batch_c : list
        list of lists, each element of batch_* is sequence (in a list format)

    Returns
    -------
    ali_batch_a : list
    ali_batch_b : list
    ali_batch_c : list
        same format as input, each element of batch_* is an aligned sequence
    """
    for l in [batch_a, batch_b, batch_c]:
        if l is not None and not isinstance(l, list):
            raise TypeError('Only list format is allowed')

    if batch_c is not None:
        if not len(batch_a) == len(batch_b) == len(batch_c):
            raise ValueError('Inconsistent number of samples in input batches')
    if batch_c is None:
        if not len(batch_a) == len(batch_b):
            raise ValueError('Inconsistent number of samples in input batches')

    batch_size = len(batch_a)
    ali_batch_a, ali_batch_b, ali_batch_c = [], [], []
    for i in range(batch_size):
        a = batch_a[i]
        b = batch_b[i]
        c = batch_c[i] if batch_c is not None else None
        ali_a, ali_b, ali_c = align_sequences(a, b, c)
        ali_batch_a.append(ali_a)
        ali_batch_b.append(ali_b)
        ali_batch_c.append(ali_c)

    if batch_c is not None:
        return ali_batch_a, ali_batch_b, ali_batch_c
    else:
        return ali_batch_a, ali_batch_b




if __name__ == '__main__':
    a = [1, 2, 2, 3, 4, 5, 6, 7]
    b = [1, 2, 3, 4, 5, 6, 6, 7]

    opt = sb.utils.edit_distance.op_table(a, b)
    print(opt)
    ali = sb.utils.edit_distance.alignment(opt)
    print(ali)

    ali_a, ali_b = [], []
    for op, a_index, b_index in ali:
        ali_a.append(a[a_index] if a_index is not None else None)
        ali_b.append(b[b_index] if b_index is not None else None)

    a_str = '|'
    b_str = '|'
    for item_a, item_b in zip(ali_a, ali_b):
        if item_a is None:
            item_a = '*'
        if item_b is None:
            item_b = '*'
        a_str += '{:^3}|'.format(item_a)
        b_str += '{:^3}|'.format(item_b)
    print(a_str)
    print(b_str)

    # wer_details = sb.utils.edit_distance.wer_details_by_utterance({'seq': a}, {'seq': b}, compute_alignments=True)
    # sb.dataio.wer.print_alignments(wer_details)