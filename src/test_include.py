import sys
import json

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

from ruamel.yaml.comments import TaggedScalar, CommentedMap


if __name__ == '__main__':
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    print(json.dumps(hparams, indent=4))