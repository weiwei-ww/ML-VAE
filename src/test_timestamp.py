import hyperpyyaml
from hyperpyyaml import load_hyperpyyaml

yaml_content = '''
timestamp: !apply:datetime.datetime.now
formatted_timestamp: !apply:datetime.datetime.strftime [!ref <timestamp>, '%y_%m%d_%H%M%S']

test_1: !ref <formatted_timestamp>
test_2: !ref results/<formatted_timestamp>
'''

loaded_yaml = load_hyperpyyaml(yaml_content)
