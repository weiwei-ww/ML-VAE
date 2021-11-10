import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

if __name__ == '__main__':
    with open('/home/weiwei/research/codes/MD-VAE-SpeechBrain/src/config/run.yaml') as f:
        hparams = load_hyperpyyaml(f, 'dataset: TIMIT')
    print(hparams)