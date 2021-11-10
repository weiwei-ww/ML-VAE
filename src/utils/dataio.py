import speechbrain as sb


def data_io_prep(hparams):
    'Creates the datasets and their data processing pipelines.'

    # 1. Define datasets:
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

    # 2. Define audio pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, sb.dataio.dataio.read_audio, takes='wav_path', provides='sig')

    # 3. Define text pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, lambda p: label_encoder.encode_sequence_torch(p), takes='phonemes', provides='encoded_phonemes')

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ['id', 'sig', 'encoded_phonemes'])

    # 5. Fit encoder:
    phoneme_set = hparams['prepare']['phoneme_set_handler'].get_phoneme_set()

    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    label_encoder.insert_blank(index=hparams['blank_index'])

    return train_dataset, valid_dataset, test_dataset, label_encoder