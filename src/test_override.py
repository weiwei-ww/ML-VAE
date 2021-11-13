import sys
import json
import ruamel.yaml

from hyperpyyaml import load_hyperpyyaml
from hyperpyyaml.core import recursive_update
import speechbrain as sb
import speechbrain

yaml_content = '''
k1: !include:/home/weiwei/research/codes/MD-VAE-SpeechBrain/src/config/ext_1.yaml
    name: new_name
'''

hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
overrides = ruamel.yaml.YAML().load(overrides)
extra_overrides = overrides.pop('extra_overrides', {})
yaml = load_hyperpyyaml(yaml_content, overrides=overrides)
print(json.dumps(yaml, indent=4))
print(json.dumps(extra_overrides))

print('-' * 100)

recursive_update(yaml, extra_overrides)
print(json.dumps(yaml, indent=4))
