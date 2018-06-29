import skorch.callbacks
import torch

class HistorySaver(skorch.callbacks.Callback):
    def __init__(self, target='history.json'):
        super().__init__()
        self.target = target

    def on_epoch_end(self, net, **kwargs):
        net.save_history(self.target.format(
            net=net,
            last_epoch=net.history[-1],
            last_batch=net.history[-1, 'batches', -1],
        ))

class OptimizerSaver(skorch.callbacks.Callback):
    def __init__(self, target='optimizer.pt'):
        super().__init__()
        self.target = target

    def on_epoch_end(self, net, **kwargs):
        torch.save(net.optimizer_.state_dict(), self.target.format(
            net=net,
            last_epoch=net.history[-1],
            last_batch=net.history[-1, 'batches', -1],
        ))
