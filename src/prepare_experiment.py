import importlib
import ruamel.yaml

from hyperpyyaml import load_hyperpyyaml
from hyperpyyaml.core import recursive_update
import speechbrain as sb

from utils.data_io import prepare_datasets

def prepare_experiment(args, prepare_exp_dir):
    # parse command line args
    hparams_file, run_opts, overrides = sb.parse_arguments(args)

    # load extra overrides
    overrides = ruamel.yaml.YAML().load(overrides)
    if overrides:
        extra_overrides = overrides.pop('extra_overrides', {})
    else:
        overrides = {}
        extra_overrides = {}

    # load hparams
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, [extra_overrides, overrides])
    recursive_update(hparams, extra_overrides)


    # create experiment directory
    if prepare_exp_dir:
        sb.create_experiment_directory(
            experiment_directory=hparams['output_dir'],
            hyperparams_to_save=hparams_file,
            overrides=[extra_overrides, overrides]
        )

    # json file preparation
    dataset_name = hparams['dataset']
    importlib.import_module(f'datasets.{dataset_name}.prepare').prepare(**hparams['prepare'])


    # Load parsed dataset
    datasets, label_encoder = prepare_datasets(hparams)

    # initialize model
    model_class = hparams['model_class']
    SBModel = importlib.import_module(f'models.{model_class}.model').SBModel
    model = SBModel(
        label_encoder=label_encoder,
        modules=hparams['model']['modules'],
        opt_class=hparams['model']['optimizer'],
        hparams=hparams['model'],
        run_opts=run_opts,
        checkpointer=hparams['model']['checkpointer'],
    )

    prepared = {
        'hparams': hparams,
        # 'hparams_file': hparams_file,
        # 'overrides': overrides,
        'datasets': datasets,
        'model': model
    }

    return prepared