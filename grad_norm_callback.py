from pytorch_lightning import Callback

class GradNormCallback(Callback):
    def on_after_backward(self, trainer, model):
        log_dict = {}
        for idx, p in enumerate(model.parameters()):
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                log_dict[f"grad_norm/layer_{idx}"] = param_norm
        model.log_dict(log_dict, prog_bar=False, on_step=True)