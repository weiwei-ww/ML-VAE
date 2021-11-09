from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

if __name__ == '__main__':

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(['model_hparams/CRDNN-CTC.yaml'])

    with open('hparams/CRDNN-CTC.yaml') as f:
        hparams = load_hyperpyyaml(f, overrides)
