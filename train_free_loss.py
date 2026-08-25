# coding: utf-8
"""
Training script for BalancingLossFreeGate on WikiText-103.
Configured for Medium Transformer-XL on a single T4 GPU (16 GB).

Usage:
    python train_free_loss.py \
        --cuda \
        --data /path/to/wt103 \
        --dataset wt103
"""
import argparse
import time
import math
import os
import sys
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from custom_gate import BalancingLossFreeGate
from new_utils import (
    set_top_k, set_router_mode, freeze_part_weight,
    set_threshold, collect_top_k,
)

import warnings
warnings.filterwarnings(action='ignore')

# ---------------------------------------------------------------------------
# Arguments — Medium Transformer-XL + BalancingLossFreeGate defaults
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Train Medium Transformer-XL with BalancingLossFreeGate on WikiText-103'
)

# Data
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103'],
                    help='dataset name (only wt103 supported)')

# Model — Medium Transformer-XL
parser.add_argument('--n_layer', type=int, default=16,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=41,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension (-1 = same as d_model)')
parser.add_argument('--d_model', type=int, default=410,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=2100,
                    help='inner dimension in FF / per-expert hidden dim')

# Regularisation
parser.add_argument('--dropout', type=float, default=0.1,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')

# Initialisation
parser.add_argument('--init', default='normal', type=str)
parser.add_argument('--emb_init', default='normal', type=str)
parser.add_argument('--init_range', type=float, default=0.1)
parser.add_argument('--emb_init_range', type=float, default=0.01)
parser.add_argument('--init_std', type=float, default=0.02)
parser.add_argument('--proj_init_std', type=float, default=0.01)

# Optimiser
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'])
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--mom', type=float, default=0.0)
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'])
parser.add_argument('--warmup_step', type=int, default=0)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--lr_min', type=float, default=0.0)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--clip_nonemb', action='store_true')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper step limit')
parser.add_argument('--eta_min', type=float, default=0.0)

# Batching — T4-safe defaults
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size (kept small for T4 16GB)')
parser.add_argument('--batch_chunk', type=int, default=2,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=150,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=150,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0)
parser.add_argument('--mem_len', type=int, default=150,
                    help='length of the retained previous heads')

# Embedding / softmax
parser.add_argument('--not_tied', action='store_true')
parser.add_argument('--adaptive', action='store_true', default=True,
                    help='use adaptive softmax (default True for wt103)')
parser.add_argument('--div_val', type=int, default=1)
parser.add_argument('--pre_lnorm', action='store_true')
parser.add_argument('--sample_softmax', type=int, default=-1)

# MoE — BalancingLossFreeGate
parser.add_argument('--moe', action='store_true', default=True,
                    help='use MoE (always True for this script)')
parser.add_argument('--moe-num-expert', type=int, default=16,
                    help='number of experts in MoE')
parser.add_argument('--moe-top-k', type=int, default=2,
                    help='top_k experts in gate')
parser.add_argument('--moe_index', type=str, default=None,
                    help='comma-separated MoE layer indices (None = all)')

# Misc
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--varlen', action='store_true')
parser.add_argument('--same_length', action='store_true')
parser.add_argument('--attn_type', type=int, default=0)
parser.add_argument('--clamp_len', type=int, default=-1)
parser.add_argument('--log-interval', type=int, default=200)
parser.add_argument('--eval-interval', type=int, default=4000)
parser.add_argument('--work_dir', default='LM-TFM-FreeLoss', type=str)
parser.add_argument('--restart', action='store_true')
parser.add_argument('--restart_dir', type=str, default='')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--max_eval_steps', type=int, default=-1)
parser.add_argument('--patience', type=int, default=0)

# FP16
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--static-loss-scale', type=float, default=1)
parser.add_argument('--dynamic-loss-scale', action='store_true')

# Freeze (gate only — no HyperRouter freezing needed)
parser.add_argument('--freeze_gate', action='store_true')
parser.add_argument('--freeze_main_network', action='store_true')
parser.add_argument('--freeze_main_network_all', action='store_true')

args = parser.parse_args()
args.tied = not args.not_tied

# Fixed: always use BalancingLossFreeGate
args.gate_name = 'BalancingLossFreeGate'
# Not used but kept for compatibility with new_utils
args.dense_drop = False
args.expert_drop = 0.5
args.num_expert = args.moe_num_expert

assert args.moe_num_expert >= args.moe_top_k, \
    "must have moe-num-expert >= moe-top-k"

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(
    args.work_dir,
    scripts_to_save=['train_free_loss.py', 'mem_transformer.py'],
    debug=args.debug,
)

# Reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# FP16 validation
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except Exception:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)

# adaptive softmax
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset == 'wt103'
    cutoffs = [20000, 40000, 200000]
    tie_projs += [True] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.moe_index is not None:
    moe_index = list(map(int, args.moe_index.split(',')))
else:
    moe_index = None

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(
        ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
        moe=args.moe, moe_num_expert=args.moe_num_expert,
        moe_top_k=args.moe_top_k,
        gate_name=BalancingLossFreeGate,   # pass class directly
        moe_index=moe_index,
        dense_drop=False, expert_drop=0.5,
        num_expert=args.moe_num_expert, attn_moe=False,
    )
    model.apply(weights_init)
    model.word_emb.apply(weights_init)

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

# Compatibility stubs
set_threshold(model, args)
freeze_part_weight(model, args)

print(model)
print("Total of Params: ", sum(p.numel() for p in model.parameters()))
print("Total of Trainable Params: ",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

if args.fp16:
    model = model.half()

para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=args.mom,
        )
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )

#### scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_step, eta_min=args.eta_min,
    )
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sparse, args.max_step, eta_min=args.eta_min,
        )
elif args.scheduler == 'inv_sqrt':
    def lr_lambda(step):
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.decay_rate, patience=args.patience,
        min_lr=args.lr_min,
    )
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sparse, factor=args.decay_rate,
            patience=args.patience, min_lr=args.lr_min,
        )
elif args.scheduler == 'constant':
    pass

if args.cuda and args.fp16:
    optimizer = FP16_Optimizer(
        optimizer,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
        dynamic_loss_args={'init_scale': 2 ** 16},
    )

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

def evaluate(model, eval_iter):
    """Evaluate the model on a given iterator."""
    model.eval()

    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len


def train():
    """Run one epoch of training."""
    global train_step, train_loss, best_val_loss, best_val_loss_dense
    global eval_start_time, log_start_time, all_top_k
    model.train()

    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    for batch, (data, target, seq_len) in enumerate(train_iter):

        current_top_k = collect_top_k(model)
        all_top_k.append(current_top_k)

        model.zero_grad()

        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            # No auxiliary balancing loss needed — BalancingLossFreeGate
            # handles balance via EMA memory, not via loss term.
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # Finalize EMA step for BalancingLossFreeGate
        if hasattr(model, 'balancing_state') and model.balancing_state is not None:
            model.balancing_state.finalize_step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 1:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            # Dense evaluation (all experts)
            current_gate = set_router_mode(model, args, flag=True)
            val_loss_dense = evaluate(model, va_iter)
            # Sparse evaluation (top-k)
            current_gate = set_router_mode(model, args, flag=False)
            val_loss = evaluate(model, va_iter)

            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)

            log_str_dense = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| Dense valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss_dense)
            log_str_dense += ' | valid ppl {:9.3f}'.format(math.exp(val_loss_dense))
            logging(log_str_dense)
            logging('-' * 100)

            # Save best sparse model
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # Save best dense model
            if not best_val_loss_dense or val_loss_dense < best_val_loss_dense:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model_dense.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer_dense.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss_dense = val_loss_dense

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
train_step = 0
train_loss = 0
best_val_loss = None
best_val_loss_dense = None
log_start_time = time.time()
eval_start_time = time.time()
all_top_k = []

try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')


# ---------------------------------------------------------------------------
# Final evaluation on test set — Dense model
# ---------------------------------------------------------------------------
with open(os.path.join(args.work_dir, 'model_dense.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

for gate_number in [1, 2, 4, 8, 16]:
    if gate_number <= args.moe_num_expert:
        set_top_k(model, gate_number)
        test_loss = evaluate(model, te_iter)
        logging('=' * 100)
        logging('Dense | End of training | Gate-Number {:.0f} '
                '| test loss {:5.2f} | test ppl {:9.3f}'.format(
                    gate_number, test_loss, math.exp(test_loss)))
        logging('=' * 100)

# ---------------------------------------------------------------------------
# Final evaluation on test set — Sparse (best val) model
# ---------------------------------------------------------------------------
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

for gate_number in [1, 2, 4, 8, 16]:
    if gate_number <= args.moe_num_expert:
        set_top_k(model, gate_number)
        test_loss = evaluate(model, te_iter)
        logging('=' * 100)
        logging('| End of training | Gate-Number {:.0f} '
                '| test loss {:5.2f} | test ppl {:9.3f}'.format(
                    gate_number, test_loss, math.exp(test_loss)))
        logging('=' * 100)

all_top_k = np.array(all_top_k)
print('* Mean Top-K During Training = {}-[{}]'.format(all_top_k, all_top_k))
