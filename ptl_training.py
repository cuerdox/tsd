"""
when it's Ray time -> https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
https://skeptric.com/pyarrow-sentencetransformers/
"""
import os

import torch
if not callable(getattr(torch, '_set_deterministic', None)):  # dirty hack to run on my custom build of old torch on laptop gpu
    torch._set_deterministic = torch.set_deterministic
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from ptl_tsd_model import TokenSenseDisambiguationModel
from prepare_data import TRAIN_CORPORA, TEST_CORPORA, get_dataset_name_for_corpora, get_hf_dataset, get_cached_model_vocab
from grad_norm_callback import GradNormCallback

from set_os_env import set_os_env
set_os_env()

def train(model_name):
    train_corpora = TRAIN_CORPORA
    dataset = get_hf_dataset(get_dataset_name_for_corpora(model_name, train_corpora))
    batch_size = 2048
    shared_params = {
        # "num_workers": 0,
        "num_workers": os.cpu_count(),
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
    }
    train_dataloader = DataLoader(dataset['train'], shuffle=True, **shared_params)
    val_dataloader = DataLoader(dataset['test'], shuffle=False, **shared_params)

    token_sense_vocabulary, reverse_token_sense_vocabulary, all_token_list = get_cached_model_vocab(model_name)
    # ckpt = None
    ckpt = "best.ckpt"
    if os.path.exists(ckpt):
        print("Resuming from ckeckpoint: " + ckpt)
        model = TokenSenseDisambiguationModel.load_from_checkpoint(
            checkpoint_path=ckpt,
            map_location=None,
            config={
                # "lr": 1e-3,
                # "emb_lr": 1e-3,
                'lr_schedule_gamma': 0.5,
                'lr_schedule_patience': 3,
                # logging params
                # 'log_embeddings': 1,
                # 'log_conf_mat': 1,
                # 'log_miscls': 1,
            }
        )
    else:
        model_config = _get_model_config(model_name, batch_size)
        model = TokenSenseDisambiguationModel(
            config=model_config,
            vocab=all_token_list,
            token_sense_vocabulary=token_sense_vocabulary)

    checkpoint_callback = ModelCheckpoint(
        # dirpath="chkpts/",
        save_top_k=2,
        monitor="hp_metric",
        mode="max",
        # every_n_val_epochs=10,
        # filename=model_params + "-{epoch:03d}-{val_loss:.4f}",
        filename="tsd-{epoch:03d}-{hp_metric:.5f}",
        verbose=True
    )

    es_callback = EarlyStopping(
        monitor="hp_metric",
        # min_delta=0.0,
        patience=50,
        verbose=True,
        mode="max",
        # strict=True,
        # check_finite=True,
        # stopping_threshold=0.01,
        # divergence_threshold=None,
        # check_on_train_epoch_end=None
    )

    trainer = pl.Trainer(
        # profiler='pytorch',
        # limit_train_batches=1,
        # limit_val_batches=1,
        # max_epochs=3000,
        max_epochs=1000,
        # track_grad_norm=2,
        #gradient_clip_val=100,
        # gradient_clip_algorithm="value",
        gpus=1 if torch.cuda.is_available() else 0,
        # logger=None
        callbacks=[
            checkpoint_callback,
            GradNormCallback(),
            LearningRateMonitor(),
            es_callback
        ]
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    test(model_name, trainer)
    # trainer.save_checkpoint("last.ckpt")
    print("are we there yet?")

def _get_model_config(model_name, batch_size):
    config = {
        # model params
        'emb_dim': 200,  # MAX_SENSE_COUNT,  # equal to number of classes
        'emb_max_norm': 0,  # change it back to None?
        'emb_mode': 'combine',  # 'combine' or 'multiply_elem' or 'multiply_filter'
        'num_layers': 1,
        'front_factor': 1.25,
        'hip_factor': 1.25,
        'dropout_p': 0.15,
        'activation': 'relu',
        'batch_size': batch_size,
        # optimization params
        'ce_weights': '0', #'1', #'0', 'balanced'
        'lr': 3e-3,
        'emb_lr': 6e-4,
        'optim': 'adam',  # 'adamW'
        'lr_schedule': [10, 200],
        'lr_schedule_gamma': 0.3,
        # logging params
        'log_embeddings': 0,
        'log_conf_mat': 0,
        'log_miscls': 0,
        # tune
        'model_name': model_name,
        #
        'default_hparams': False
    }
    return config


def test(model_name, trainer=None):
    test_corpora = TEST_CORPORA
    dataset_names = get_dataset_name_for_corpora(model_name, test_corpora, type='test')
    shared_params = {
        # "num_workers": 0,
        "num_workers": os.cpu_count(),
        "batch_size": 1,  # yes, 'train' split and batch_size of 1 - limitation of current code - variable length of tokens per example
    }
    test_dataloaders = [DataLoader(get_hf_dataset(ds_name)['test'], **shared_params) for ds_name in dataset_names]

    if trainer is not None:
        results = trainer.test(ckpt_path="best", dataloaders=test_dataloaders)
    else:
        trainer = pl.Trainer(
            # gpus=1 if torch.cuda.is_available() else 0,  # more overhead with gpu than profit with batch_size of 1
        )
        model = TokenSenseDisambiguationModel.load_from_checkpoint(
            checkpoint_path="best.ckpt",
            map_location=None,
        )
        results = trainer.test(model=model, dataloaders=test_dataloaders)

    print(results)
    print("Tested! Are you happy now?")


if __name__ == '__main__':
    train('distilbert-base-uncased')
    # test('distilbert-base-uncased')
