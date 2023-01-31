import torch
import torch.cuda
from torch.utils.data import DataLoader

import ray
from ray import air, tune

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ptl_tsd_model import TokenSenseDisambiguationModel
from prepare_data import TEST_CORPORA, get_hf_dataset, get_dataset_name_for_corpora

import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]="gloo"

def test_model(model_name):
    ray.init(
        object_store_memory=128 * 1024 * 1024,
        # local_mode=True,
    )

    test_corpora = TEST_CORPORA

    config = {
        'model_name': model_name,
        'corpus': tune.grid_search(test_corpora)
    }

    gpus_per_trial = 0 if not torch.cuda.is_available() else 0.25  # can you?
    cpus_per_trial = 1  # default value

    model = TokenSenseDisambiguationModel.load_from_checkpoint(
        checkpoint_path="best.ckpt",
        map_location=None,
    )

    train_fn_with_parameters = tune.with_parameters(_tune_model_testing_fn, model=model)
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="test/acc_0_all",
            mode="max",
        ),
        run_config=air.RunConfig(
            local_dir="./ray_results/",
            name="test_model",
            # progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()
    df = results.get_dataframe()
    print(df)
    return df

def _tune_model_testing_fn(config, model):
    if not callable(getattr(torch, '_set_deterministic',
                            None)):  # dirty hack to run on my custom build of old torch on laptop gpu
        torch._set_deterministic = torch.set_deterministic

    model_name = config['model_name']
    corpus = config['corpus']

    dataset_names = get_dataset_name_for_corpora(model_name, [corpus], type='test')
    test_dataloaders = \
        [DataLoader(get_hf_dataset(ds_name)['test'], batch_size=1, num_workers=2) for ds_name in dataset_names]

    trainer = pl.Trainer(
        # limit_test_batches=100,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False,
    )
    results = trainer.test(model=model, dataloaders=test_dataloaders)
    tune.report(**results[0])


if __name__ == '__main__':
    test_model('distilbert-base-uncased')
