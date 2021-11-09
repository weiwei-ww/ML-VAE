import os
import json
import logging
from pathlib import Path

from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

from utils.phonemes import map_phoneme

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare(
    dataset_dir,
    train_json_path,
    valid_json_path,
    test_json_path,
    n_phonemes
):
    '''
    Parse the json files for the TIMIT dataset.

    Arguments
    ---------
    dataset_dir : str
        Path to the dir where the original TIMIT dataset is stored.

    Example
    -------
    >>> from recipes.TIMIT.timit_prepare import prepare_timit
    >>> dataset_dir = 'datasets/TIMIT'
    >>> prepare_timit(dataset_dir, 'train.json', 'valid.json', 'test.json')
    '''
    # initialization
    dataset_dir = Path(dataset_dir)

    train_json_path = Path(train_json_path)
    valid_json_path = Path(valid_json_path)
    test_json_path = Path(test_json_path)
    json_paths = [train_json_path, valid_json_path, test_json_path]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # check if it is a valid TIMIT directory and if it is uppercase
    if (dataset_dir / 'TRAIN' / 'DR1').is_dir() and (dataset_dir / 'TEST' / 'DR1').is_dir():
        uppercase = True
    elif (dataset_dir / 'train' / 'dr1').is_dir() and (dataset_dir / 'test' / 'dr1').is_dir():
        uppercase = False
    else:
        raise FileNotFoundError(f'not a valid TIMIT directory: {dataset_dir.absolute()}')

    # check if this step is already performed, and skip it if so
    skip = True
    for json_path in [train_json_path, valid_json_path, test_json_path]:
        if not json_path.exists():
            skip = False
    if skip:
        logger.info('skip preparation')
        return

    # create json files
    logger.info('create json files for the TIMIT dataset')

    sub_dirs = ['train', 'test', 'test']
    extensions = ['.wav']
    dev_spks, test_spks = _get_speaker()
    avoid_sentences = ['sa1', 'sa2']
    if uppercase:
        sub_dirs = [item.upper() for item in sub_dirs]
        extensions = [item.upper() for item in extensions]
        dev_spks = [item.upper() for item in dev_spks]
        test_spks = [item.upper() for item in test_spks]
        avoid_sentences = [item.upper() for item in avoid_sentences]

    json_paths = [train_json_path, valid_json_path, test_json_path]
    match_ors = [None, dev_spks, test_spks]
    invalid_suffixes = ['.WAV.wav']

    for sub_dir, json_path, match_or in zip(sub_dirs, json_paths, match_ors):
        match_and = extensions + [sub_dir]

        # get wav files
        wav_lst = get_all_files(
            dataset_dir,
            match_and=match_and,
            match_or=match_or,
            exclude_and=invalid_suffixes,
            exclude_or=avoid_sentences,
        )

        # create json file
        create_json(wav_lst, json_path, uppercase, n_phonemes)


def create_json(
    wav_lst, json_file, uppercase, n_phonemes,
):
    '''
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
            The path of the output json file.
    uppercase : bool
        Whether this is the uppercase version of timit.
    n_phonemes : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
    '''

    # Adding some Prints
    msg = 'Creating %s...' % (json_file)
    logger.info(msg)
    json_dict = {}

    for wav_file in wav_lst:

        # Getting sentence and speaker ids
        spk_id = wav_file.split('/')[-2]
        if uppercase:
            snt_id = wav_file.split('/')[-1].replace('.WAV', '')
        else:
            snt_id = wav_file.split('/')[-1].replace('.wav', '')
        snt_id = spk_id + '_' + snt_id

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = len(signal) / SAMPLERATE

        # Retrieving words and check for uppercase
        if uppercase:
            wrd_file = wav_file.replace('.WAV', '.WRD')
        else:
            wrd_file = wav_file.replace('.wav', '.wrd')

        if not os.path.exists(os.path.dirname(wrd_file)):
            err_msg = 'the wrd file %s does not exists!' % (wrd_file)
            raise FileNotFoundError(err_msg)

        # words = [line.rstrip('\n').split(' ')[2] for line in open(wrd_file)]
        # words = ' '.join(words)
        words, word_segments = get_phoneme_lists(wrd_file, n_phonemes=-1)


        # Retrieving phonemes
        if uppercase:
            phn_file = wav_file.replace('.WAV', '.PHN')
        else:
            phn_file = wav_file.replace('.wav', '.phn')

        if not os.path.exists(os.path.dirname(phn_file)):
            err_msg = 'the phn file %s does not exists!' % (phn_file)
            raise FileNotFoundError(err_msg)

        # Getting the phoneme and ground truth ends lists from the phn files
        phonemes, phoneme_segments = get_phoneme_lists(phn_file, n_phonemes)

        json_dict[snt_id] = {
            'wav_path': wav_file,
            'duration': duration,
            'spk_id': spk_id,
            'canonicals': phonemes,
            'phonemes': phonemes,
            'phoneme_segments': phoneme_segments,
            'words': words,
            'word_segments': word_segments
        }

    # Writing the dictionary to the json file
    with open(json_file, mode='w') as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f'{json_file} successfully created!')


def _get_speaker():

    # List of test speakers
    test_spk = [
        'fdhc0',
        'felc0',
        'fjlm0',
        'fmgd0',
        'fmld0',
        'fnlp0',
        'fpas0',
        'fpkt0',
        'mbpm0',
        'mcmj0',
        'mdab0',
        'mgrt0',
        'mjdh0',
        'mjln0',
        'mjmp0',
        'mklt0',
        'mlll0',
        'mlnt0',
        'mnjm0',
        'mpam0',
        'mtas1',
        'mtls0',
        'mwbt0',
        'mwew0',
    ]

    # List of dev speakers
    dev_spk = [
        'fadg0',
        'faks0',
        'fcal1',
        'fcmh0',
        'fdac1',
        'fdms0',
        'fdrw0',
        'fedw0',
        'fgjd0',
        'fjem0',
        'fjmg0',
        'fjsj0',
        'fkms0',
        'fmah0',
        'fmml0',
        'fnmr0',
        'frew0',
        'fsem0',
        'majc0',
        'mbdg0',
        'mbns0',
        'mbwm0',
        'mcsh0',
        'mdlf0',
        'mdls0',
        'mdvc0',
        'mers0',
        'mgjf0',
        'mglb0',
        'mgwt0',
        'mjar0',
        'mjfc0',
        'mjsw0',
        'mmdb1',
        'mmdm2',
        'mmjr0',
        'mmwh0',
        'mpdf0',
        'mrcs0',
        'mreb0',
        'mrjm4',
        'mrjr0',
        'mroa0',
        'mrtk0',
        'mrws1',
        'mtaa0',
        'mtdt0',
        'mteb0',
        'mthc0',
        'mwjg0',
    ]

    return dev_spk, test_spk


def get_phoneme_lists(phn_file, n_phonemes):
    '''
    Reads the phn file and gets the phoneme list & ground truth ends list.
    '''

    phonemes = []
    segments = []

    for line in open(phn_file):
        start, end, phoneme = line.rstrip('\n').replace('h#', 'sil').split(' ')

        # Removing end corresponding to q if phn set is not 61
        # if n_phonemes != 60:
        #     if phoneme == 'q':
        #         phoneme = ''

        # Converting phns if necessary
        if n_phonemes != -1:
            phoneme = map_phoneme(phoneme)

        # Appending arrays
        if len(phoneme) > 0:
            phonemes.append(phoneme)
            segments.append([int(start) / SAMPLERATE, int(end) / SAMPLERATE])

    if n_phonemes != 60 and n_phonemes != -1:
        # Filtering out consecutive silences by applying a mask with `True` marking
        # which sils to remove
        # e.g.
        # phonemes          [  'a', 'sil', 'sil',  'sil',   'b']
        # ends              [   1 ,    2 ,    3 ,     4 ,    5 ]
        # ---
        # create:
        # remove_sil_mask   [False,  True,  True,  False,  False]
        # ---
        # so end result is:
        # phonemes ['a', 'sil', 'b']
        # ends     [  1,     4,   5]

        remove_sil_mask = [True if x == 'sil' else False for x in phonemes]

        for i, val in enumerate(remove_sil_mask):
            if val is True:
                if i == len(remove_sil_mask) - 1:
                    remove_sil_mask[i] = False
                elif remove_sil_mask[i + 1] is False:
                    remove_sil_mask[i] = False

        phonemes = [
            phon for i, phon in enumerate(phonemes) if not remove_sil_mask[i]
        ]
        segments = [segment for i, segment in enumerate(segments) if not remove_sil_mask[i]]
        assert len(phonemes) == len(segments), f'{len(phonemes)} phonemes, {len(segments)} segments'
        for i, (phoneme, _) in enumerate(zip(phonemes, segments)):
            phoneme = phonemes[i]
            if phoneme != 'sil':
                continue
            if i == 0:  # first sil always has a zero start time
                segments[i][0] = 0.0
            else:  # the start time of sil should be the end time of the previous phoneme
                segments[i][0] = segments[i - 1][1]


    return phonemes, segments