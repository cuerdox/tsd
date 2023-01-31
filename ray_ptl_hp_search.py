"""
https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
"""
import torch
from torch.utils.data import DataLoader

if not callable(getattr(torch, '_set_deterministic', None)):  # dirty hack to run on my custom build of old torch on laptop gpu
    torch._set_deterministic = torch.set_deterministic

import pytorch_lightning as pl
import ray
import torch.cuda

from prepare_data import MAX_SENSE_COUNT, get_cached_model_vocab, get_hf_dataset, get_dataset_name_for_corpora
from ptl_tsd_model import TokenSenseDisambiguationModel
from grad_norm_callback import GradNormCallback

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

# tune search algorithms
# from ray.tune.search.bayesopt import BayesOptSearch
# Nevergrad
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
# flaml
# from ray.tune.search.flaml import BlendSearch
# from ray.tune.search.flaml import CFO

import os
import math

TUNE_LOCAL_DIR="./ray_results/"

def tune_model_grid():
    config = {
        # model params
        'emb_dim': tune.grid_search([1, 64, MAX_SENSE_COUNT * 2]),
        'emb_mode': tune.grid_search(['combine', 'multiply_elem']),  # 'combine' or 'multiply_elem' or 'multiply_filter'
        # 'ctx_dim': 768,
        'batch_norm': tune.grid_search([True, False]),
        'num_layers': tune.grid_search([0, 2, 5]),
        'front_factor': tune.grid_search([0.5, 1, 1.5]),
        'hip_factor': tune.grid_search([1, 1.5]),
        'activation': tune.grid_search(['relu', 'gelu']), # , 'elu', 'tanh']),
        # dataset params
        'batch_size': tune.grid_search([64, 2048]),
        # optimization params
        # 'ce_weights': tune.grid_search(['0', '1', 'balanced']),
        'ce_weights': 0,
        'lr': 1e-3,
        'emb_lr': 5e-3,
        'optim': 'adam',
        # logging params
        'log_embeddings': 0,
        'log_conf_mat': 0,
        'log_miscls': 0,
        # tune
        'model_name': 'distilbert-base-uncased',
        #
        'default_hparams': False
    }

    gpus_per_trial = 0 if not torch.cuda.is_available() else 0.5  # can you?
    cpus_per_trial = 2  # default value

    num_epochs = 5

    # reporter = CLIReporter(
    #     parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
    #     metric_columns=["loss", "mean_accuracy", "training_iteration"])

    trainable_with_parameters = _get_trainable_with_parameters(config["model_name"],
                                                               False,  # enable_checkpointing
                                                               num_epochs,
                                                               cpus_per_trial,
                                                               gpus_per_trial)

    tuner = tune.Tuner(
        trainable_with_parameters,
        tune_config=tune.TuneConfig(
            metric="hp_metric",
            mode="max",
        ),
        run_config=air.RunConfig(
            local_dir=TUNE_LOCAL_DIR,
            name="tune_tsd_model_grid",
            # progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_config = results.get_best_result().config
    print("Best hyperparameters found were: ", best_config)
    return best_config

def tune_model_asha():
    config = {
        # model params
        # 'emb_dim': tune.lograndint(1, MAX_SENSE_COUNT * 2),
        'emb_dim': tune.choice([1, 64, MAX_SENSE_COUNT]),
        # 'emb_max_norm': tune.randint(0, 100),
        'emb_max_norm': tune.choice([0, 1, 4, 16]),
        # 'ctx_dim': 768,
        'num_layers': tune.choice([1, 5]),
        # 'front_factor': tune.quniform(lower=0.25, upper=2, q=0.25),
        'front_factor': tune.choice([1, 1.25]),
        # 'hip_factor': tune.quniform(lower=1, upper=2, q=0.25),
        'hip_factor': tune.choice([1, 1.25]),
        'dropout_p': tune.choice([0, 0.1, 0.2, 0.3]),
        'activation': tune.choice(['relu', 'lrelu', 'gelu', 'elu', 'tanh']),
        # dataset params
        # 'dataset_name': None,
        # 'dataset_name': 'ds_distilbert-base-uncased_semcor',
        # 'dataset_pcent': 10,
        # 'dataset_keep_in_memory': False,
        'batch_size': tune.choice([64, 384, 768, 2048]),
        # optimization params
        'ce_weights': tune.choice(['0', '1', 'balanced']),
        'lr': tune.loguniform(1e-5, 1e-1),
        'emb_lr': tune.loguniform(1e-5, 1e-1),
        'optim': tune.choice(['adam', 'adamW']),
        # logging params
        'log_embeddings': 0,
        'log_conf_mat': 0,
        'log_miscls': 0,
        # tune
        'model_name': 'distilbert-base-uncased',
        #
        'default_hparams': False
    }

    num_samples = 50

    gpus_per_trial = 0 if not torch.cuda.is_available() else 0.5  # can you?
    cpus_per_trial = 2  # default value


    num_epochs = 5
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        # grace_period=1,
        reduction_factor=2)

    # search algorithms
    # algo = BayesOptSearch(random_search_steps=16, verbose=3)  # can't handle categorical values?
    # algo = CFO()
    # algo = BlendSearch()
    search_algo = NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne,
        metric="hp_metric",
        mode="max",
        points_to_evaluate=None)

    # reporter = CLIReporter(
    #     parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
    #     metric_columns=["loss", "mean_accuracy", "training_iteration"])

    trainable_with_parameters = _get_trainable_with_parameters(config["model_name"],
                                                               False,  # enable_checkpointing
                                                               num_epochs,
                                                               cpus_per_trial,
                                                               gpus_per_trial)

    tuner = tune.Tuner(
        trainable_with_parameters,
        tune_config=tune.TuneConfig(
            metric="hp_metric",
            mode="max",
            scheduler=scheduler,
            search_alg=search_algo,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            local_dir=TUNE_LOCAL_DIR,
            name="tune_tsd_model_asha",
            # progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_config = results.get_best_result().config
    print("Best hyperparameters found were: ", best_config)
    return best_config


def tune_model_pbt(config, num_samples=4, num_epochs=100):
    if config is None:
        config = {
            # model params
            'emb_dim': 200,
            'emb_max_norm': 0,
            # 'ctx_dim': 768,
            'num_layers': 2,
            'front_factor': 1,
            # 'hip_factor': tune.quniform(lower=1, upper=2, q=0.25),
            'hip_factor': 1,
            'dropout_p': tune.uniform(0.1, 0.4),
            'activation': 'relu',
            # dataset params
            'dataset_name': None,
            # 'dataset_name': 'ds_distilbert-base-uncased_semcor',
            # 'dataset_pcent': 10,
            # 'dataset_keep_in_memory': False,
            'batch_size': tune.choice([2048, 4096, 8192]),
            # optimization params
            'ce_weights': tune.choice(['0', '1', 'balanced']),
            "lr": tune.loguniform(5e-3, 1e-2),
            "emb_lr": tune.loguniform(5e-3, 1e-1),
            'optim': tune.choice(['adam', 'adamW']),
            # logging params
            'log_embeddings': 0,
            'log_conf_mat': 0,
            'log_miscls': 0,
            # tune
            'model_name': 'distilbert-base-uncased',
            #
            'default_hparams': False
        }

    gpus_per_trial = 0 if not torch.cuda.is_available() else 0.5
    cpus_per_trial = 2  # default value

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        # perturbation_interval=4,  # every 10 `time_attr` units (training_iterations in this case)
        perturbation_interval=1,  # every 10 `time_attr` units (training_iterations in this case)
        burn_in_period=0,
        resample_probability=0.33,
        perturbation_factors=(2, 0.5),
        hyperparam_mutations={
            'optim': ['adam', 'adamW'],
            'dropout_p': tune.uniform(0.01, 0.4),
            'ce_weights': ['0', '1', 'balanced'],
            "lr": tune.loguniform(1e-4, 1e-1),
            "emb_lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([64, 256, 512, 2048, 8192])
        })

    # reporter = CLIReporter(
    #     parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
    #     metric_columns=["loss", "mean_accuracy", "training_iteration"])

    trainable_with_parameters = _get_trainable_with_parameters(config["model_name"],
                                                               False,  # enable_checkpointing
                                                               num_epochs,
                                                               cpus_per_trial,
                                                               gpus_per_trial)

    tuner = tune.Tuner(
        trainable_with_parameters,
        tune_config=tune.TuneConfig(
            metric="hp_metric",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            local_dir=TUNE_LOCAL_DIR,
            name="tune_tsd_model_pbt",
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="hp_metric",
                checkpoint_score_order="max"
            ),
            failure_config=air.FailureConfig(
                max_failures=-1,
                fail_fast=False,
            ),
            # progress_reporter=reporter,
            verbose=3,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


def _get_vocab_and_ds_refs(model_name):
    # put vocab into object store
    _, _, token_list = get_cached_model_vocab(model_name)
    vocab_ref = ray.put(token_list)

    # put dataset into object store
    ds_ref = None
    train_corpora = [
        ("data/semcor.xml.gz", 9e5, "complete_tagging", "paragraph", 4),
        ("data/wngt.xml.gz", 1.8e6, "partial_tagging", "sentence", 32),
        ## ("data/omtsi.xml", int(1e7), "partial_tagging", "?", "?")
    ]
    dataset_name = get_dataset_name_for_corpora(model_name, train_corpora, type='train')
    dataset = get_hf_dataset(dataset_name, keep_in_memory=False, pcent=100)
    ds_ref = ray.put(dataset)

    return vocab_ref, ds_ref

def _get_dataloaders_for_dataset(dataset, batch_size):
    train_dataloader, val_dataloader = None, None
    if dataset is not None:
        ray_local_mode = ray.worker._mode() == 2  # dirty hack, don't do it like that again
        num_workers = os.cpu_count() if not ray_local_mode else 0
        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
        val_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_dataloader, val_dataloader

def _get_tune_callback(enable_checkpointing, checkpoint_filename):
    tune_callback_params = {
        "metrics": {
            "1_f1": "val/f1",
            "2_precision": "val/precision",
            "3_recall": "val/recall",
            "4_acc": "val/acc",
            "5_loss": "val/loss",
            "hp_metric": "hp_metric",
        },
        "on": "validation_end"
    }
    if enable_checkpointing:
        tune_callback_params["filename"]=checkpoint_filename
        tune_callback = TuneReportCheckpointCallback(**tune_callback_params)
    else:
        tune_callback = TuneReportCallback(**tune_callback_params)
    return tune_callback

def _get_trainable_with_parameters(model_name, enable_checkpointing, num_epochs, cpus_per_trial, gpus_per_trial):
    vocab_ref, ds_ref = _get_vocab_and_ds_refs(model_name)
    trainable_with_parameters = tune.with_resources(
            tune.with_parameters(
                launch_trainer,
                vocab=vocab_ref,
                dataset=ds_ref,
                enable_checkpointing=enable_checkpointing,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial),
        resources={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        )
    return trainable_with_parameters


def launch_trainer(config, vocab, dataset, enable_checkpointing=False, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    # dirty hack to run on my custom build of old torch on laptop gpu
    if not callable(getattr(torch, '_set_deterministic', None)):
        torch._set_deterministic = torch.set_deterministic

    # suppress warnings inside ray trainable
    import logging
    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", category=Warning)


    CHECKPOINT_FILENAME = "checkpoint"

    kwargs = {
        "max_epochs": num_epochs,
        # "limit_train_batches": 5,  # debug
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        "enable_checkpointing": enable_checkpointing,
        "enable_progress_bar": False,
        "callbacks": [
            GradNormCallback(),
            _get_tune_callback(enable_checkpointing, CHECKPOINT_FILENAME)
        ],
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    trainer = pl.Trainer(**kwargs)
    model = TokenSenseDisambiguationModel(config=config, vocab=vocab)
    train_dataloader, val_dataloader = _get_dataloaders_for_dataset(dataset, config["batch_size"])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def restore_tune(experiment_name, num_epochs=100, cpus_per_trial=2, gpus_per_trial=0.5):
    model_name='distilbert-base-uncased'
    # Reconstruct the trainable with the same parameters
    trainable_with_params = _get_trainable_with_parameters(model_name=model_name,
                                                           enable_checkpointing=experiment_name.endswith("pbt"),
                                                           num_epochs=num_epochs,
                                                           cpus_per_trial=cpus_per_trial,
                                                           gpus_per_trial=gpus_per_trial)

    tuner = tune.Tuner.restore(path=os.path.join(TUNE_LOCAL_DIR, experiment_name),
                               restart_errored=True,
                               overwrite_trainable=trainable_with_params)

    results = tuner.fit()
    print("Experiment results (best config): ", results.get_best_result().config)


if __name__ == '__main__':
    ray.init(
        object_store_memory=128 * 1024 * 1024,
        # local_mode=True,
    )

    # tune_model_grid()
    best_config = tune_model_asha()
    # tune_model_pbt(config=best_config)
    # tune_model_pbt(config=None)
    # restore_tune("tune_tsd_model_grid", num_epochs=5, gpus_per_trial=1)
    # restore_tune("tune_tsd_model_asha")
    # restore_tune("tune_tsd_model_asha")
