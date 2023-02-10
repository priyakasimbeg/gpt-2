from torcheval.metrics.text import Perplexity
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics import Metric

class PerplexityIgnite(Metric):

    def __init__(self, ignore_index=None, device='cpu', output_transform=lambda x: x):
        self.perplexity = Perplexity(ignore_index=ignore_index, device=device)
        super(PerplexityIgnite, self).__init__(output_transform=output_transform, device=device)

    def __name__(self):
        return "CustomPerplexity"

    @reinit__is_reduced
    def reset(self):
        self.perplexity.reset()
        super(PerplexityIgnite, self).reset()

    @reinit__is_reduced
    def update(self, output):
        input=output[0].detach()
        target=output[1].detach()
        self.perplexity.update(input, target)

    def compute(self):
        return self.perplexity.compute()