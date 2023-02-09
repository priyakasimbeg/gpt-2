# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license.
""" A very small and self-contained gist to train a GPT-2 transformer model on wikitext-103 """
import os
from collections import namedtuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup, ProgressBar
from pytorch_pretrained_bert import BertTokenizer, cached_path
from torcheval.metrics.text import Perplexity
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics import Metric
from vocab import Corpus

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


class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        """ Transformer (GPT-2 architecture) """
        super().__init__()
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                 	nn.ReLU(),
                                                 	nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout)
        self.lm_head = nn.Linear(config.embed_dim, config.num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.tokens_embeddings.weight  # Tie weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss

        return logits

      
Config = namedtuple('Config',
    field_names="embed_dim, hidden_dim, num_max_positions, num_embeddings, num_heads, num_layers," 
                "dropout, initializer_range, batch_size, lr, max_norm, n_epochs, n_warmup, device,"
                "gradient_accumulation_steps, log_dir, dataset_cache, dataset_valid_cache",
    defaults   =[410      , 2100      , 256              , 50000         , 10        , 16         ,
                 0.1    , 0.02             , 16         , 2.5e-4, 0.25, 200     , 1000    , "cuda",
                 4                          , "./"   , "./dataset_cache_small_gist_tokenized", "./dataset_cache_small_gist_valid_tokenized"])

# Load a pre-defined tokenizer (BERT), create config and model
args = Config()

if not os.path.isfile(args.dataset_cache) or not os.path.isfile(args.dataset_valid_cache):
    dataset_path = os.path.expanduser("~/init2winit/wikitext-103/gpt-2/data/wikitext-103")
    print("Training tokenizer.")
    tokenized_corpus = Corpus(dataset_path)

# Tokenize wikitext-103 training dataset
if os.path.isfile(args.dataset_cache):
    dataset = torch.load(args.dataset_cache)
else:
    dataset = tokenized_corpus.train
    torch.save(dataset, args.dataset_cache)
if os.path.isfile(args.dataset_valid_cache):
    valid_dataset = torch.load(args.dataset_valid_cache)
else:
    valid_dataset = tokenized_corpus.valid
    torch.save(valid_dataset, args.dataset_valid_cache)

args = Config(num_embeddings=len(tokenized_corpus.dictionary.word2idx.keys()), device="cuda" if torch.cuda.is_available() else "cpu")
model = TransformerWithLMHead(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Organize the dataset in blocs of num_max_positions tokens for the transformer
num_sequences = (dataset.size(0) // args.num_max_positions) * args.num_max_positions
dataset = dataset.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
train_eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# Organize the dataset in blocs of num_max_positions tokens for the transformer
num_valid_sequences = (valid_dataset.size(0) // args.num_max_positions) * args.num_max_positions
valid_dataset = valid_dataset.narrow(0, 0, num_valid_sequences).view(-1, args.num_max_positions)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

# Define training function
def update(engine, batch):
    model.train()
    batch = batch.transpose(0, 1).contiguous().to(args.device)  # to shape [seq length, batch]
    logits, loss = model(batch, labels=batch)
    loss = loss / args.gradient_accumulation_steps
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    if engine.state.iteration % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()
trainer = Engine(update)

# Add progressbar with loss
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

# Evaluation function and evaluator (evaluator output is the input of the metrics)
def inference(engine, batch):
    model.eval()
    with torch.no_grad():
        batch = batch.transpose(0, 1).contiguous().to(args.device)  # to shape [seq length, batch]
        labels = batch
        logits, loss = model(batch, labels=labels)
        shift_logits = logits[:-1]
        shift_labels = labels[1:]
    return shift_logits, shift_labels
valid_evaluator = Engine(inference)
train_evaluator = Engine(inference)


# Attache metric to evaluator & evaluation to trainer: evaluate on valid set after each epoch
PerplexityIgnite().attach(valid_evaluator, 'perplexity')
PerplexityIgnite().attach(train_evaluator, 'perplexity')
@trainer.on(Events.ITERATION_COMPLETED(every=1000))
def log_validation_results(engine):
    valid_evaluator.run(valid_loader, max_epochs=128, epoch_length=1)
    train_evaluator.run(train_eval_loader, max_epochs=128, epoch_length=1)

    batch_loss = engine.state.output
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    v_p = valid_evaluator.state.metrics['perplexity']
    t_p = train_evaluator.state.metrics['perplexity']
    print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, train_perplexity: {t_p}, valid_perplexity: {v_p}")
    # print(f"Perplexity: {evaluator.state.metrics['perplexity']}")
    # # print(f"Validation Epoch: {engine.state.epoch} Error rate: {evaluator.state.metrics['perplexity']}")

# Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero with cosine
cos_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, len(dataloader) * args.n_epochs)
scheduler = create_lr_scheduler_with_warmup(cos_scheduler, 0.0, args.n_warmup, args.lr)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

# Save checkpoints and training config
checkpoint_handler = ModelCheckpoint(args.log_dir, 'checkpoint', save_interval=1, n_saved=5)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': model})
torch.save(args, os.path.join(args.log_dir, 'training_args.bin'))

trainer.run(dataloader, max_epochs=args.n_epochs)