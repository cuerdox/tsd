import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import *

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

from prepare_data import MAX_SENSE_COUNT, get_hf_dataset


class TokenSenseDisambiguationModel(pl.LightningModule):
    def __init__(self, *args, config=None, vocab=None, token_sense_vocabulary=None, **kwargs):
        super().__init__()

        if vocab is None:
            vocab = ['-'] * 30522  # default size for Bert-based tokenizers
        self.vocab = vocab

        self.token_sense_vocabulary = token_sense_vocabulary

        self.default_config = {
            # model params
            'num_senses': MAX_SENSE_COUNT,
            'emb_num': len(vocab),
            'emb_dim': MAX_SENSE_COUNT,  # equal to number of classes
            'emb_max_norm': 0,
            'emb_mode': 'combine',  # 'combine' or 'multiply_elem' or 'multiply_filter'
            'num_filters': 3,
            'ctx_dim': 768,
            'batch_norm': False,
            'num_layers': 3,
            'front_factor': 2,
            'hip_factor': 1.5,
            'dropout_p': 0.2,
            'activation': 'relu',
            'act_multi': None,
            # dataset params
            'dataset_name': None,
            'dataset_pcent': 100,
            'dataset_keep_in_memory': False,
            'batch_size': 2048,
            # optimization params
            'ce_weights': '0',
            'lr_scheduler': 'cos',
            'lr': 1e-3,
            'emb_lr': 1e-3,
            'optim': 'adam',
            'multi_optimizer': True,
            'lr_schedule': None,
            'lr_schedule_gamma': 0.5,
            'lr_schedule_patience': 3,
            # logging params
            'log_embeddings': 0,
            'log_conf_mat': 0,
            'log_miscls': 0,
            # tune
            'model_name': 'distilbert-base-uncased',
            #
            'default_hparams': True
        }
        active_config = self.default_config.copy()

        if len(kwargs) > 0:
            active_config.update(kwargs)
        if config is not None:  # order matters here: allow config object override kwargs which are received when loading checkpoint
            active_config.update(config)
        self.save_hyperparameters(active_config)

        emb_num = active_config["emb_num"]
        emb_dim = active_config["emb_dim"]
        emb_max_norm = active_config["emb_max_norm"]
        emb_max_norm = None if emb_max_norm == 0 else emb_max_norm
        ctx_dim = active_config["ctx_dim"]

        emb_mode = active_config["emb_mode"]
        self.emb_mode = emb_mode
        if emb_mode == 'combine':
            input_size = emb_dim + ctx_dim
            self.embeddings = nn.Embedding(emb_num, emb_dim, padding_idx=0, max_norm=emb_max_norm, sparse=False)
        elif emb_mode == 'multiply_elem':
            if emb_dim != self.default_config["emb_dim"]:
                raise NotImplementedError
            emb_dim = ctx_dim  # for now just hard code same size
            input_size = ctx_dim
            self.embeddings = nn.Embedding(emb_num, emb_dim, padding_idx=0, sparse=False)
            torch.nn.init.uniform_(self.embeddings.weight[1:], 0.98, 1.02)  # should be fine for element-wise multiplication
        elif emb_mode == 'multiply_filter':
            self.num_filters = active_config["num_filters"]
            emb_dim = ctx_dim * self.num_filters
            input_size = self.num_filters
            self.embeddings = nn.Embedding(emb_num, emb_dim, padding_idx=0, sparse=False)
            torch.nn.init.uniform_(self.embeddings.weight[1:], -1e-4, 1e-4)  # should be fine for matrix multiplication

        output_size = active_config["num_senses"]

        batch_norm = active_config["batch_norm"]
        ff = active_config["front_factor"]
        num_layers = active_config["num_layers"]
        hip_factor = active_config["hip_factor"]
        dropout_p = active_config["dropout_p"]

        configured_activation = active_config['activation']
        configured_multi_activation = active_config['act_multi']

        hidden_size = int(input_size * ff)
        hip_size = int(hip_factor * hidden_size)

        self.layers = nn.Sequential(*self._get_layers(num_layers,
                                                      input_size,
                                                      output_size,
                                                      hidden_size,
                                                      hip_size,
                                                      batch_norm,
                                                      dropout_p,
                                                      configured_activation,
                                                      configured_multi_activation))

        ce_weights = active_config['ce_weights']
        if ce_weights == '0':  # decrease 0-th to 0.1, all the rest to 1
            ce_weights = torch.ones((MAX_SENSE_COUNT,))
            ce_weights[0] = 0.1
        elif ce_weights == '1':  # no weighting
            ce_weights = torch.ones((MAX_SENSE_COUNT,))
        elif ce_weights == 'balanced':  # balanced, computed in jupyter across full semcor and wngt datasets
            ce_weights = torch.tensor([2.96297741e-02, 9.62050504e-02, 1.25376665e-01, 1.59170914e-01,
       1.74814710e-01, 2.15712981e-01, 2.51988711e-01, 2.74848312e-01,
       3.19815125e-01, 3.63684284e-01, 3.79071254e-01, 4.03538858e-01,
       4.19216844e-01, 4.68365780e-01, 5.31829458e-01, 5.33095199e-01,
       6.51400421e-01, 7.05601238e-01, 6.33671349e-01, 7.40426878e-01,
       7.92649786e-01, 7.24165552e-01, 8.27485987e-01, 9.52458211e-01,
       8.67605850e-01, 1.02829903e+00, 9.67928029e-01, 9.46214858e-01,
       1.21370796e+00, 1.14847330e+00, 1.35469552e+00, 1.31573121e+00,
       1.44158123e+00, 1.52482746e+00, 1.79738932e+00, 1.70090731e+00,
       1.74899435e+00, 1.93039079e+00, 2.26491109e+00, 2.01481545e+00,
       2.38464207e+00, 2.44569465e+00, 2.84901974e+00, 2.18418460e+00,
       2.60038631e+00, 2.25234570e+00, 2.64701100e+00, 3.75694910e+00,
       3.25113363e+00, 3.07128369e+00, 3.82553887e+00, 3.13653549e+00,
       4.61674840e+00, 3.76129994e+00, 3.30069360e+00, 5.45404282e+00,
       5.14312352e+00, 4.72763100e+00, 7.06829706e+00, 6.45061072e+00,
       8.69580321e+00, 7.48360023e+00, 6.68288580e+00, 1.00865916e+01,
       8.24335660e+00, 1.36505800e-01, 1.67515924e-01, 3.18123561e-01,
       4.35518941e-01, 5.71056264e-01, 6.67121803e-01, 7.72018659e-01,
       9.76659901e-01, 1.24202008e+00, 1.79938089e+00, 1.64491390e+00,
       2.05822719e+00, 2.34419524e+00, 2.28241918e+00, 2.55437082e+00,
       3.13653549e+00, 3.46810731e+00, 2.55638135e+00, 4.67657667e+00,
       4.37425253e+00, 5.09871664e+00, 2.12210552e+00, 3.02409916e+00,
       4.20982826e+00, 7.03766522e+00, 5.70304214e+00, 7.16181367e+00,
       8.54705921e+00, 7.17764088e+00, 7.24165552e+00, 6.14547304e+00,
       1.10660392e+01, 7.01486501e+00, 7.35647225e+00, 4.35372989e+00,
       1.51770210e+01, 1.37331184e+01, 1.29655988e+01, 6.22199713e+00,
       1.15995804e+01, 1.13166638e+01, 1.39994935e+01, 2.05562184e+01,
       2.12279902e+01, 2.62986437e+01, 2.51773837e+01, 1.44030266e+01,
       2.43287079e+01, 2.06214762e+01, 2.89989509e+01, 1.40297300e+01,
       1.10660392e+01, 2.79989871e+01, 1.17890472e+01, 2.81201948e+01,
       1.34210021e+01, 3.51122432e+01, 3.12296394e+01, 2.56749605e+01,
       2.65133265e+01, 3.45519415e+01, 3.45519415e+01, 2.93926018e+01,
       4.16395192e+01, 2.37071715e+01, 2.56749605e+01, 1.89934649e+01,
       4.19081613e+01, 3.96083232e+01, 3.23172388e+01, 4.24559804e+01,
       6.30656796e+01, 1.28884226e+01, 1.91615487e+01, 5.07481641e+01,
       1.41828930e+01, 3.02128605e+01, 7.21751667e+01, 3.69077557e+01,
       5.36840083e+01, 7.29861236e+01, 8.32790385e+01, 2.69533817e+01,
       7.29861236e+01, 7.29861236e+01, 8.32790385e+01, 9.99348462e+01,
       1.08262750e+02, 6.98469355e+01, 1.10097712e+02, 1.29915300e+02,
       1.03107381e+02, 8.11970625e+01, 1.22561604e+02, 1.41212283e+02,
       1.85593286e+02, 9.96130195e-02, 1.95879772e-01, 3.16959354e-01,
       4.20138736e-01, 4.73866720e-01, 7.29123920e-01, 9.53436812e-01,
       8.55268598e-01, 1.59994212e+00, 1.75989298e+00, 2.14736033e+00,
       2.00362893e+00, 2.15591271e+00, 2.96339644e+00, 3.42965417e+00,
       3.65752534e+00, 4.57126319e+00, 5.09871664e+00, 5.75355624e+00,
       5.43578661e+00, 5.70805360e+00, 5.66326504e+00, 8.41420337e+00,
       2.31083778e-01, 4.42551097e-01, 1.59758116e+00, 2.12071988e+00,
       4.16929718e+00, 5.65833188e+00, 3.91311145e+00, 3.95119526e+00,
       1.03766214e+01, 1.02456861e+01, 1.51416434e+01, 2.31166014e+01,
       8.08937111e+00, 2.33660612e+01, 1.10849232e+01, 4.08538679e+01])

        self.loss_fn = nn.CrossEntropyLoss(weight=ce_weights)

        # dataset initialization for ptl_run in tune
        if active_config['dataset_name'] is not None:
            dataset_name = active_config['dataset_name']
            dataset_pcent = active_config['dataset_pcent']
            keep_in_memory = active_config['dataset_keep_in_memory']
            self.dataset = get_hf_dataset(dataset_name, keep_in_memory=keep_in_memory, pcent=dataset_pcent)
            self.batch_size = active_config['batch_size']

        self._graph_logged = False
        self._best_val_loss = 100
        self._best_val_epoch = 0
        self._epochs_since_best_val = 0


    def _get_layers(self, num_layers, input_size, output_size, hidden_size, hip_size, batch_norm, dropout_p,
                    configured_activation, configured_multi_activation):
        activations = {
            'relu': nn.ReLU,
            'relu6': nn.ReLU6,
            'lrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'rrelu': nn.RReLU,
            'selu': nn.SELU,  # read the docs!
            'celu': nn.CELU,  # read the docs!
            'gelu': nn.GELU,  # read the docs!
            'elu': nn.ELU,
            'tanh': nn.Tanh,
            'multi': 'multi'
        }
        activation = activations[configured_activation]

        def _wrap_multi(act_func_name, idx):
            if isinstance(act_func_name, str):
                return activations[configured_multi_activation[idx]]()
            else:
                return act_func_name()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_size))
        if num_layers == 0:
            layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(input_size, hip_size))
            layers.append(_wrap_multi(activation, 0))
            for i in range(num_layers - 1):
                layers.append(nn.Linear(hip_size if i==0 else hidden_size, hidden_size))
                layers.append(_wrap_multi(activation, i + 1))
                # if i > 1 and 1 % 3 == 0:
                #     layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(hip_size if num_layers == 1 else hidden_size, output_size))

        return layers

    def forward(self, inputs):
        # like in hyperbolic - glue together token_id and ctx_emb -> 1; 768 => 769 and in forward unpack back
        # expected dims: batch * (1+ctx_hidd_dim) to turn into: batch * emb_dim|ctx_hidden_dim

        # split glued token_id and ctx_hidden, get token_emb for token_id and glue all together again
        batched_token_ids = inputs.narrow(-1, 0, 1).to(torch.long)  # select first element in a batch
        batched_token_embs = self.embeddings(batched_token_ids).squeeze(dim=1)  # batch x seq_len x emb_dim

        ctx_hidden_size = inputs.size(-1) - 1
        batched_ctx_hiddens = inputs.narrow(-1, 1, ctx_hidden_size)  # select all except first (ctx_hidden)

        data_item = None
        if self.emb_mode == 'combine':
            data_item = torch.hstack((batched_token_embs, batched_ctx_hiddens))
        elif self.emb_mode == 'multiply_elem':
            data_item = batched_ctx_hiddens * batched_token_embs
        elif self.emb_mode == 'multiply_filter':
            data_item = torch.bmm(batched_ctx_hiddens.unsqueeze(1),
                                  batched_token_embs.view(-1, ctx_hidden_size, self.num_filters)).squeeze(1)

        pred = self.layers(data_item)
        return pred


    def get_optimizer_parameter_groups_with_lr(self):
        # see for inspiration https://github.com/Lightning-AI/lightning/issues/7576
        embedding = self.embeddings.parameters()
        cls = self.layers.parameters()
        lr = self.hparams.lr
        emb_lr = self.hparams["emb_lr"] or lr
        optimizer_parameters = [
            {"params": embedding, "name": "embeddings", "lr": emb_lr},
            {"params": cls, "name":"cls", "lr": lr}
        ]
        return optimizer_parameters


    def configure_optimizers(self):
        def get_optimizer(ps):
            if not isinstance(ps, list):
                ps = [ps]
            if self.hparams.optim == 'adam':
                optimizer = optim.Adam(params=ps)
            elif self.hparams.optim == 'adamW':
                optimizer = optim.AdamW(params=ps)
            else:
                raise 'Optimizer is not identified!'
            return optimizer

        def get_scheduler(opt, metric):
            scheduler_name = self.hparams['lr_scheduler']
            if scheduler_name == "step":
                scheduler_config = {
                    "scheduler": optim.lr_scheduler.MultiStepLR(opt,
                                                                milestones=self.hparams['lr_schedule'],
                                                                gamma=self.hparams['lr_schedule_gamma']),
                    "monitor": metric,
                }
            elif scheduler_name == "plato":
                scheduler_config = {
                    "scheduler": ReduceLROnPlateau(opt,
                                                   mode="max",
                                                   factor=self.hparams['lr_schedule_gamma'],
                                                   patience=self.hparams['lr_schedule_patience'],
                                                   verbose=True),
                    "monitor": metric,
                }
            elif scheduler_name == "cos":
                scheduler_config = {
                    "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                            T_0=4,
                                            T_mult=1,
                                            eta_min=1e-6),
                    "monitor": metric,
                }
            else:
                raise 'Scheduler is not identified!'

            return scheduler_config

        params = self.get_optimizer_parameter_groups_with_lr()

        optimizer_list = []
        lr_schedulers = []
        if self.hparams["multi_optimizer"]:
            for p in params:
                optimizer = get_optimizer(p)
                metric = "val/" + ("recall" if p["name"] == "embeddings" else "precision")
                lr_schedulers.append(get_scheduler(optimizer, metric))
                optimizer_list.append(optimizer)
        else:
            optimizer = get_optimizer(params)
            lr_schedulers.append(get_scheduler(optimizer, "val/f1"))
            optimizer_list.append(optimizer)

        return optimizer_list, lr_schedulers

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # # debug
        # from torch.utils.data import TensorDataset
        # dataset = get_tensor_dataset(model_name_or_path='distilbert-base-uncased', corpus_filename='semcor.xml')
        # return DataLoader(TensorDataset(*dataset['train']), batch_size=self.batch_size, shuffle=True)
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # # debug
        # from torch.utils.data import TensorDataset
        # dataset = get_tensor_dataset(model_name_or_path='distilbert-base-uncased', corpus_filename='semcor.xml')
        # return DataLoader(TensorDataset(*dataset['val']), batch_size=self.batch_size, shuffle=True)
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=2)

    def _unpack_batch(self, batch):
        if type(batch) is dict:  # came from hf dataset
            src = batch['merged-token_id-ctx_value']
            tgt = batch['token_sense_id']
        else:  # came from mine dataset
            src, tgt = batch
        return src, tgt

    def _unpack_test_batch(self, batch):
        # ['ex_id', 'ss_id', 'token_lst', 'token_id_lst', 'ctx_lst']
        src = torch.hstack((batch['token_id_lst'].T, torch.squeeze(batch['ctx_lst'],dim=0)))
        tgt = batch['ss_id'][0]
        tokens = batch['token_lst']
        # in torch.data.utils.collate default_collate line zip(*batch) turns list of list of 1 into a tuple of 1, fixing it here
        tokens = [item if isinstance(item, list) else item[0] for item in tokens]

        return src, tgt, tokens

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        src, tgt = self._unpack_batch(batch)
        pred = self(src)
        loss = self.loss_fn(pred, tgt)

        if not self._graph_logged:
            self.logger.experiment.add_graph(TokenSenseDisambiguationModel().to(self.device), src)
            self._graph_logged = True
        log_dict, full_dict = self._prepare_step_metrics(src, tgt, pred, loss, 'train')
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False)
        return full_dict

    def validation_step(self, batch, batch_idx):
        src, tgt = self._unpack_batch(batch)
        pred = self(src)
        loss = self.loss_fn(pred, tgt)
        
        log_dict, full_dict = self._prepare_step_metrics(src, tgt, pred, loss, 'val')
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return full_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        src, tgt, tokens = self._unpack_test_batch(batch)
        logits = self(src)
        predicted_sense = self._gather_sense(tokens, logits, tgt)

        log_dict, full_dict = self._prepare_test_step_metrics(tgt, predicted_sense, dataloader_idx=dataloader_idx)
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return full_dict

    def _gather_sense(self, tokens, logits, tgt):
        cls = F.softmax(logits, dim=-1)

        if len(tokens) > 1:
            # 'sum' probs for each sense among all tokens excluding 0-th sense
            # idea is that if sense will be present among multiple tokens
            # sum of it's per-token probabilities will be high enough to win among all
            # unaligned sense_ids for each token makes life harder here
            top_k_senses = torch.topk(cls, 3, dim=-1)
            candidates = {}
            tokens_top_k = [token_top_k for token_top_k in zip(*top_k_senses, tokens)]
            idx_val_tokens = [zip(token_top_k[1].tolist(), token_top_k[0].tolist(), [token_top_k[2]]*len(tokens))
                              for token_top_k in tokens_top_k]
            ss_val_list = [[(self._get_token_sense(t, i), v) for i, v, t in i_v_token] for i_v_token in idx_val_tokens]
            [[candidates.update({ss_id: candidates.get(ss_id, 0) + val}) for ss_id, val in ss_val_item if ss_id != 0]
             for ss_val_item in ss_val_list]
            if (len(candidates) == 0):
                print("Debug: empty candidates for tokens: ", tokens)
            predicted_sense = max(candidates.items(), key=lambda x: x[1])[0] if len(candidates) > 0 else 0
        else:
            predicted_sense_id = torch.argmax(cls, dim=-1).tolist()
            predicted_sense = [self._get_token_sense(token, sense_id) for token, sense_id in zip(tokens, predicted_sense_id)][0]

        # Debug logging goes here?
        # print(final_sense == tgt, tgt, final_sense, ", ".join(map(str,predicted_senses)))
        return predicted_sense

    def _get_token_sense(self, token, sense_id):
        empty_dict = {}  # for tokens that were never encountered during vocab building but can pop in testing
        ss_id = 0 if sense_id == 0 else self.token_sense_vocabulary.get(token, empty_dict).get(sense_id, 0)
        return ss_id

    def _prepare_step_metrics(self, src, tgt, pred, loss, tag):
        src, tgt, pred = [t.clone().detach().cpu() for t in (src, tgt, pred)]
        pred_sense_id = torch.argmax(pred.data, -1)
        accuracy = ((pred_sense_id == tgt).sum() / float(pred_sense_id.shape[0]))
        misclassified_token_ids = self._get_misclassified_token_ids(src, pred_sense_id, tgt)
        log_dict = {
            'loss': loss,  # PTL wants it
            tag + '/loss': loss.item(),
            tag + '/acc': accuracy.detach(),
        }
        full_dict = {
            tag + '/predicted_sense_id': pred_sense_id.detach(),
            tag + '/gold_sense_id': tgt.detach(),
            tag + '/miscls': misclassified_token_ids,
        }
        full_dict.update(log_dict)
        return log_dict, full_dict

    def _prepare_test_step_metrics(self, tgt, pred, dataloader_idx=0, tag='test'):
        pos = tgt.split('.')[1]
        if pos == 's':  # putting together 'Adjectives' and 'Adjectives satellite'
            pos = 'a'
        sorting_order = {'n': 1, 'v': 2, 'a': 3, 'r': 4}

        accuracy = float(pred == tgt)
        # misclassified_senses = (tgt, pred)

        # from collections import defaultdict
        # if not hasattr(self, "per_pos_test_res"):
        #     self.per_pos_test_res = defaultdict(lambda : defaultdict(list))
        # self.per_pos_test_res[dataloader_idx][pos].append(accuracy)

        log_dict = {
            tag + '/acc_0_all': accuracy,
            tag + f'/acc_{sorting_order[pos]}_{pos}': accuracy,
        }
        full_dict = {
            # tag + '/miscls': misclassified_senses,
        }
        full_dict.update(log_dict)
        return log_dict, full_dict


    def _get_metric(self, outputs, metric_name, fn):
        return fn([next(v for k, v in metric_dict.items() if k == metric_name) for metric_dict in outputs])

    def training_epoch_end(self, outputs):
        if isinstance(outputs[0], list) and len(outputs) > 1:
            merged_output = []
            [merged_output.extend(output) for output in outputs]
            outputs = merged_output

        self._log_epoch_metrics('train', outputs)


    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val', outputs)


    def test_epoch_end(self, outputs):
        # for dl in self.per_pos_test_res.keys():
        #     for k, v in self.per_pos_test_res[dl].items():
        #         print(f"Dataloader: {dl} Pos: {k} len: {len(v)} acc: {sum(v)/len(v)}")
        ...

    def _log_epoch_metrics(self, tag, outputs):
        loss_var, loss_mean = self._get_metric(outputs, tag + '/loss', lambda x: torch.var_mean(torch.tensor(x), unbiased=False))
        acc = self._get_metric(outputs, tag + '/acc', lambda x: torch.mean(torch.tensor(x)))
        preds = self._get_metric(outputs, tag + '/predicted_sense_id', torch.cat)
        golds = self._get_metric(outputs, tag + '/gold_sense_id', torch.cat)

        precision, recall, f1, _ = precision_recall_fscore_support(golds, preds, average='macro', zero_division=0)
        prfs_metrics = {
            tag + '/loss': loss_mean,
            tag + '/loss_var': loss_var,
            tag + '/acc': acc,
            tag + '/precision': precision,
            tag + '/recall': recall,
            tag + '/f1': f1,
        }
        self.log_dict(prfs_metrics, prog_bar=True, logger=True)

        if tag == 'train':
            lrs = {optimizer.param_groups[0]["name"] + " lr": optimizer.param_groups[0]['lr'] for optimizer in self.optimizers()}
            self.log_dict(lrs, prog_bar=True, logger=False)

        if tag == 'val':
            self.log("hp_metric", f1)  # for Tensorboard PTL way

            loss_mean = loss_mean if not torch.isnan(loss_mean) else 1e5
            if loss_mean < self._best_val_loss:
                self._best_val_loss = loss_mean
                self._best_val_epoch = self.current_epoch
                self._epochs_since_best_val = 0
            else:
                self._epochs_since_best_val += 1
            best_epoch = {
                'val/best_val_loss': float(self._best_val_loss),
                'val/best_val_epoch': float(self._best_val_epoch),
                'val/epochs_since_best_val': float(self._epochs_since_best_val),
            }
            self.log_dict(best_epoch)

        if isinstance(self.logger, TensorBoardLogger):  # TensorBoard specific logging
            if self.hparams.log_miscls and self.current_epoch % self.hparams.log_miscls == 0:
                self._log_misclassified_tokens(tag + ' misclassified tokens: ',
                                               self._get_metric(outputs, tag + '/miscls', lambda x: x))
            if tag == 'val':
                self.logger.experiment.add_histogram('predicted_sense_ids', preds, self.current_epoch, bins=self.hparams["num_senses"])
                if self.hparams.log_conf_mat and self.current_epoch % self.hparams.log_conf_mat == 0:
                    self._log_confusion_matrix(preds, golds)
                if self.hparams.log_embeddings and self.current_epoch % self.hparams.log_embeddings == 0:
                    self._log_embeddings()

    def _get_misclassified_token_ids(self, src, pred, tgt):
        mask = pred != tgt
        miscls = set()
        if not torch.all(mask == False):
            token_ids = (src.narrow(-1, 0, 1).to(torch.long)[mask]).squeeze(-1).tolist()
            tgt_sense_ids = tgt[mask].squeeze(-1).tolist()
            if not isinstance(tgt_sense_ids, list):
                tgt_sense_ids = [tgt_sense_ids]
            predicted_sense_ids = pred[mask].squeeze(-1).tolist()
            if not isinstance(predicted_sense_ids, list):
                predicted_sense_ids = [predicted_sense_ids]
            miscls.update(zip(token_ids, tgt_sense_ids, predicted_sense_ids))
        return miscls

    def _log_misclassified_tokens(self, tag, misclassified_tuples):
        miscls_token_tuples = set()
        [miscls_token_tuples.update(s) for s in misclassified_tuples]
        sorted_tuples = sorted(miscls_token_tuples, key=lambda x: x[0])
        miscls_tokens = f"misclassified count: {len(sorted_tuples)} " + \
                        ", ".join([f" **{self.vocab[token_id]}**--{tgt_sense_id}-->{predicted_sense_id}" for token_id, tgt_sense_id, predicted_sense_id in
                                   sorted_tuples])
        self.logger.experiment.add_text(tag, miscls_tokens, self.current_epoch)

    def _log_embeddings(self):
        # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=area%20under%20curve
        try:
            # no global_step=self.current_epoch to save space and keep only last
            self.logger.experiment.add_embedding(self.embeddings.weight, metadata=self.vocab)
        except:
            # fix for exception in tf/tb guts from https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
            import tensorflow as tf
            import tensorboard as tb
            tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

            # no global_step=self.current_epoch to save space and keep only last
            self.logger.experiment.add_embedding(self.embeddings.weight, metadata=self.vocab)

    def _log_confusion_matrix(self, preds, golds):
        # FIXME: prettify font here
        fig, ax = plt.subplots(figsize=(18, 18), dpi=96)
        _ = ConfusionMatrixDisplay.from_predictions(
            golds,
            preds,
            normalize='true',
            include_values=False,
            cmap="Blues",
            ax=ax,
            colorbar=False)
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)

        # AUC
        # self.logger.experiment.add_pr_curve_raw()
        ####
        # confusion matrix
        # https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
        # from sklearn.metrics import confusion_matrix
        # import numpy as np
        #
        # labels = ...
        # predictions = ...
        #
        # cm = confusion_matrix(labels, predictions)
        # recall = np.diag(cm) / np.sum(cm, axis=1)
        # precision = np.diag(cm) / np.sum(cm, axis=0)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['vocab'] = self.vocab
        checkpoint['token_sense_vocabulary'] = self.token_sense_vocabulary


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.vocab = checkpoint['vocab']
        self.token_sense_vocabulary = checkpoint['token_sense_vocabulary']
