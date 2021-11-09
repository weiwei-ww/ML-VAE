import argparse
import re
from pathlib import Path
from tgt.io import read_textgrid


def parse_textgrid(path, level):
    """
    Parse the TextGrid annotation file.

    Parameters
    ----------
    path : Path
        The annotation file path.

    level : str
        all, word, phoneme, or canonical_phoneme

    Returns
    -------
    annotation : dict
        Parsed annotation.

    """
    tg = read_textgrid(path)
    annotation = {
        'all': {
            'start_time': tg.start_time,
            'end_time': tg.end_time,
            'word': [],
            'phoneme': [],
            'canonical_phoneme': []
        },
        'word': [],
        'phoneme': [],
        'canonical_phoneme': []
    }

    # read word-level annotation
    tier = tg.get_tier_by_name('words')
    for interval in tier:
        start_time = interval.start_time
        end_time = interval.end_time
        text = interval.text
        annotation['word'].append(text)
        annotation['all']['word'].append((start_time, end_time, text))

    # read phone-level annotation
    tier = tg.get_tier_by_name('phones')
    for interval in tier:
        start_time = interval.start_time
        end_time = interval.end_time
        text = interval.text

        # format the phoneme
        text = text.lower().split(',')
        # get the phone name
        phoneme = text[0] if len(text) == 1 else text[1]
        canonical_phoneme = text[0]
        # format the phone name
        phoneme = re.findall('[a-zA-Z]+', phoneme)[0]
        canonical_phoneme = re.findall('[a-zA-Z]+', canonical_phoneme)[0]

        annotation['phoneme'].append(phoneme)
        annotation['all']['phoneme'].append((start_time, end_time, phoneme))
        annotation['canonical_phoneme'].append(canonical_phoneme)
        annotation['all']['canonical_phoneme'].append((start_time, end_time, canonical_phoneme))

    return annotation[level]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the TextGrid file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-textgrid', help='path to the TextGrid file', required=True)
    parser.add_argument('-level', help='word, phoneme, or canonical_phoneme',
                        choices=['word', 'phoneme', 'canonical_phoneme'], required=True)
    args = parser.parse_args()
    result = parse_textgrid(Path(args.textgrid), args.level)
    print('\n'.join(result))
