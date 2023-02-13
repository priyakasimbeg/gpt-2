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
from vocab import Tokenizer
from transformer import TransformerWithLMHead
from metrics import PerplexityIgnite
import logging
logging.basicConfig(filename='run001.log', level=logging.INFO)
from torchinfo import summary

LOG_INTERVAL=1000
CUDA =  "cuda" if torch.cuda.is_available() else "cpu"
CPU = "cpu"

Config = namedtuple('Config',
    field_names="embed_dim, hidden_dim, num_max_positions, num_embeddings, num_heads, num_layers," 
                "dropout, initializer_range, batch_size, lr, max_norm, n_epochs, n_warmup, device,"
                "gradient_accumulation_steps, log_dir, dataset_cache, dataset_valid_cache, vocab_path",
    defaults   =[410      , 2100      , 256              , 267735         , 10        , 16         ,
                 0.1    , 0.02             , 16        , 2.5e-4,       0.25, 200     , 1000    , CUDA,
                 4       , "./"   , "./dataset_cache_small_gist_tokenized", "./dataset_cache_small_gist_valid_tokenized",
                 "wikitext103.vocab"])

# Load a pre-defined tokenizer (BERT), create config and model
args = Config()

dataset_path = os.path.expanduser("data/wikitext-103")
tokenizer = Tokenizer(dataset_path)
if os.path.isfile(args.vocab_path):
    logging.info("Loading vocab")
    tokenizer.load_vocab(args.vocab_path)
else: 
    logging.info("Training tokenizer")
    tokenizer.train()
    tokenizer.save_vocab(args.vocab_path)

logging.info("Tokenizing dataset")
tokenizer.tokenize_wikitext103()

# Tokenize wikitext-103 training dataset
if os.path.isfile(args.dataset_cache):
    dataset = torch.load(args.dataset_cache)
else:
    dataset = torch.tensor(tokenizer.train_data, dtype=torch.long)
    torch.save(dataset, args.dataset_cache)
if os.path.isfile(args.dataset_valid_cache):
    valid_dataset = torch.load(args.dataset_valid_cache)
else:
    valid_dataset = torch.tensor(tokenizer.valid_data, dtype=torch.long)
    torch.save(valid_dataset, args.dataset_valid_cache)

logging.info("Making model and optimizer")
model = TransformerWithLMHead(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# print model dims
summary(model, input_size=(args.batch_size, args.num_max_positions ), dtypes=[torch.long])

logging.info("making data loaders")
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
PerplexityIgnite(device=args.device).attach(valid_evaluator, 'perplexity')
PerplexityIgnite(device=args.device).attach(train_evaluator, 'perplexity')
@trainer.on(Events.ITERATION_COMPLETED(every=LOG_INTERVAL))
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
    # logging.info(f"Perplexity: {evaluator.state.metrics['perplexity']}")
    # # logging.info(f"Validation Epoch: {engine.state.epoch} Error rate: {evaluator.state.metrics['perplexity']}")

# Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero with cosine
cos_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, len(dataloader) * args.n_epochs)
scheduler = create_lr_scheduler_with_warmup(cos_scheduler, 0.0, args.n_warmup, args.lr)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

# Save checkpoints and training config
checkpoint_handler = ModelCheckpoint(args.log_dir, 'checkpoint', save_interval=1, n_saved=5)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': model})
torch.save(args, os.path.join(args.log_dir, 'training_args.bin'))

logging.info("Started training.")
trainer.run(dataloader, max_epochs=args.n_epochs)