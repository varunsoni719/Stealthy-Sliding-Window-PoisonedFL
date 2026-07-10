import argparse
import os
import random
import yaml
import math
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from scipy.stats import binom
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY & SETUP
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    """Enforce strict determinism for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g_rng = torch.Generator()

# ─────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 30, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(30, 50, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(50 * 5 * 5, 100), nn.ReLU(),
            nn.Linear(100, 10),
        )
    def forward(self, x): return self.net(x)

class Cifar10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x): return self.net(x)

def get_model(dataset):
    return MnistCNN().to(DEVICE) if dataset == 'mnist' else Cifar10CNN().to(DEVICE)

def flatten(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten(vec, model):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[idx:idx+n].view(p.shape))
        idx += n

def model_from_flat(vec, dataset):
    m = get_model(dataset)
    unflatten(vec, m)
    return m

# ─────────────────────────────────────────────
# 3. DATASET / NON-IID SPLIT
# ─────────────────────────────────────────────
def load_dataset(name, data_dir='./data'):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    if name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
        te = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        tr = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
        te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    return tr, te

def non_iid_split(dataset, n_clients, q):
    labels    = np.array([y for _, y in dataset])
    n_classes = len(set(labels))
    client_data = defaultdict(list)
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet([q] * n_clients)
        proportions = (proportions * len(idx)).astype(int)
        proportions[-1] = max(0, len(idx) - proportions[:-1].sum())
        start = 0
        for k, cnt in enumerate(proportions):
            client_data[k].extend(idx[start:start+cnt].tolist())
            start += cnt
    return [client_data[k] for k in range(n_clients)]

# ─────────────────────────────────────────────
# 4. LOCAL TRAINING & BACKDOOR
# ─────────────────────────────────────────────
def local_train(global_flat, indices, train_dataset, cfg):
    if len(indices) == 0: return torch.zeros_like(global_flat)
    model  = model_from_flat(global_flat.clone(), cfg['dataset'])
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=cfg['batch_size'], shuffle=True, worker_init_fn=seed_worker, generator=g_rng)
    opt    = optim.SGD(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(cfg['local_epochs']):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return flatten(model) - global_flat

def apply_trigger(x, trigger_size=5, dataset='mnist'):
    x_triggered = x.clone()
    trigger_val = (1.0 - 0.1307) / 0.3081 if dataset == 'mnist' else 2.0
    x_triggered[:, :, -trigger_size:, -trigger_size:] = trigger_val
    return x_triggered

def backdoor_local_train(global_flat, indices, train_dataset, cfg, epochs=None):
    if len(indices) == 0: return torch.zeros_like(global_flat)
    model  = model_from_flat(global_flat.clone(), cfg['dataset'])
    model.train()
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=cfg['batch_size'], shuffle=True, worker_init_fn=seed_worker, generator=g_rng)
    opt    = optim.SGD(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.CrossEntropyLoss()

    trigger_size = cfg.get('trigger_size', 5)
    target_class = cfg.get('backdoor_target', 0)
    n_epochs     = epochs if epochs is not None else cfg.get('backdoor_local_epochs', cfg['local_epochs'])

    for _ in range(n_epochs):
        for x, _ in loader:  
            x = x.to(DEVICE)
            opt.zero_grad()
            x_triggered = apply_trigger(x, trigger_size, cfg['dataset'])
            target_labels = torch.full((x.size(0),), target_class, dtype=torch.long).to(DEVICE)
            
            backdoor_logits = model(x_triggered)
            loss = loss_fn(backdoor_logits, target_labels)
            loss.backward()
            opt.step()

    return flatten(model) - global_flat

def evaluate_backdoor_global(model_flat, test_dataset, cfg):
    model = model_from_flat(model_flat.clone(), cfg['dataset'])
    model.eval()

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=g_rng)
    trigger_size = cfg.get('trigger_size', 5)
    target_class = cfg.get('backdoor_target', 0)

    clean_correct = 0; backdoor_correct = 0; total = 0; total_non_target = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            total += x.size(0)

            clean_preds   = model(x).argmax(1)
            clean_correct += (clean_preds == y).sum().item()

            non_target_mask = (y != target_class)
            if non_target_mask.sum() > 0:
                x_nt        = x[non_target_mask]
                x_triggered = apply_trigger(x_nt, trigger_size, cfg['dataset'])
                bd_preds    = model(x_triggered).argmax(1)
                backdoor_correct += (bd_preds == target_class).sum().item()
                total_non_target += non_target_mask.sum().item()

    clean_acc = clean_correct / total
    bsr       = backdoor_correct / max(total_non_target, 1)
    return clean_acc, bsr, (1.0 - clean_acc)

def evaluate_backdoor_split(model_flat, dataset, indices, cfg):
    if len(indices) == 0: return 0.0
    model = model_from_flat(model_flat.clone(), cfg['dataset'])
    model.eval()

    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=g_rng)

    trigger_size = cfg.get('trigger_size', 5)
    target_class = cfg.get('backdoor_target', 0)
    backdoor_correct = 0; total_non_target = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            non_target_mask = (y != target_class)
            if non_target_mask.sum() > 0:
                x_nt        = x[non_target_mask]
                x_triggered = apply_trigger(x_nt, trigger_size, cfg['dataset'])
                bd_preds    = model(x_triggered).argmax(1)
                backdoor_correct += (bd_preds == target_class).sum().item()
                total_non_target += non_target_mask.sum().item()

    return backdoor_correct / max(total_non_target, 1)

# ─────────────────────────────────────────────
# 5. ATTACKS & DYNAMICS
# ─────────────────────────────────────────────
class CDynamics:
    def __init__(self, c0, c_min, c_max, block_size=50, round_jitter=0.03, block_change_min=0.5, block_change_max=2.0, alpha=1.02, beta=0.98, seed=0):
        self.c_min = c_min
        self.c_max = c_max
        self.block_size = max(1, block_size)
        self.round_jitter = round_jitter
        self.block_change_min = block_change_min
        self.block_change_max = block_change_max
        self.alpha = alpha
        self.beta  = beta

        self.block_base   = float(np.clip(c0, c_min, c_max))
        self._target      = self.block_base
        self.round_in_block = 0
        self.rng = np.random.RandomState(seed)

    def step(self, success: bool) -> float:
        if self.round_in_block == 0 and self._has_run_once():
            factor = self.rng.uniform(self.block_change_min, self.block_change_max)
            new_base = self._target * factor  
            self.block_base = float(np.clip(new_base, self.c_min, self.c_max))
            self._target = self.block_base

        nudge = self.alpha if success else self.beta
        self._target = float(np.clip(self._target * nudge, self.c_min, self.c_max))
        
        noise = self.rng.uniform(-self.round_jitter, self.round_jitter)
        c_t = float(np.clip(self._target * (1.0 + noise), self.c_min, self.c_max))

        self.round_in_block = (self.round_in_block + 1) % self.block_size
        return c_t

    def _has_run_once(self):
        ran = getattr(self, '_ran', False)
        self._ran = True
        return ran

class PoisonedFL:
    def __init__(self, d, cfg):
        torch.manual_seed(cfg['seed'])
        self.s       = (torch.randint(0, 2, (d,), dtype=torch.float32) * 2 - 1).to(DEVICE)
        self.c_dyn   = CDynamics(
            c0=cfg['c0'], c_min=cfg['c_min'], c_max=cfg['c_max'],
            block_size=cfg['e'], round_jitter=cfg.get('c_round_jitter', 0.03),
            block_change_min=cfg.get('c_block_change_min', 0.5), block_change_max=cfg.get('c_block_change_max', 2.0),
            alpha=cfg['alpha'], beta=cfg['beta'], seed=cfg['seed'],
        )
        self.c = cfg['c0']; self.e = cfg['e']; self.p_value = cfg['p_value']; self.d = d
        self.k_prev = None; self.v_prev = None; self.current_lambda = cfg['c0']; self.cfg = cfg

    def compute_unit_magnitude(self, g_prev, k_prev_s):
        norm_g = torch.norm(g_prev); norm_ks = torch.norm(k_prev_s)
        if norm_ks < 1e-10: return (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE)
        scaled_ks = (norm_g / norm_ks) * k_prev_s
        diff      = torch.abs(g_prev - scaled_ks)
        norm_diff = torch.norm(diff)
        if norm_diff < 1e-10: return (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE)
        return diff / norm_diff

    def hypothesis_test(self, w_history):
        if len(w_history) < self.e + 1: return True
        delta   = w_history[-1] - w_history[-self.e - 1]
        matches = (torch.sign(delta) == self.s).float().sum().item()
        p       = 1.0 - binom.cdf(int(matches) - 1, self.d, 0.5)
        return p <= self.p_value

    def craft_update(self, g_prev, w_history, n_fake):
        v_t = (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE) if self.k_prev is None or g_prev is None else self.compute_unit_magnitude(g_prev, self.k_prev * self.s)
        
        if g_prev is None:
            lambda_t = self.c_dyn.block_base
        else:
            norm_gprev = torch.norm(g_prev).item()
            if math.isnan(norm_gprev) or math.isinf(norm_gprev): norm_gprev = 1.0
            norm_gprev = min(norm_gprev, 100.0)

            succeeded  = self.hypothesis_test(w_history)
            self.c     = self.c_dyn.step(succeeded)
            lambda_t   = self.c * norm_gprev

        self.current_lambda = lambda_t
        k_t = lambda_t * v_t
        g_malicious = k_t * self.s
        self.k_prev = k_t; self.v_prev = v_t
        return [g_malicious.clone() for _ in range(n_fake)]


class SSWPoisonedFL:
    def __init__(self, d, cfg, train_dataset, malicious_train_indices):
        torch.manual_seed(cfg['seed'])
        self.s = (torch.randint(0, 2, (d,), dtype=torch.float32) * 2 - 1).to(DEVICE)
        
        self.c_dyn   = CDynamics(
            c0=cfg['c0'], c_min=cfg['c_min'], c_max=cfg['c_max'],
            block_size=cfg['e'], round_jitter=cfg.get('c_round_jitter', 0.03),
            block_change_min=cfg.get('c_block_change_min', 0.5), block_change_max=cfg.get('c_block_change_max', 2.0),
            alpha=cfg['alpha'], beta=cfg['beta'], seed=cfg['seed'],
        )
        self.c = cfg['c0']; self.e = cfg['e']; self.p_value = cfg['p_value']; self.d = d
        self.current_lambda = cfg['c0']; self.cfg = cfg
        self.train_dataset = train_dataset
        self.malicious_train_indices = malicious_train_indices
        self.last_g_base = None

    def compute_unit_magnitude(self, g_prev, actual_malicious_sent):
        norm_g = torch.norm(g_prev); norm_sent = torch.norm(actual_malicious_sent)
        if norm_sent < 1e-10: return (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE)
        scaled_sent = (norm_g / norm_sent) * actual_malicious_sent
        diff      = torch.abs(g_prev - scaled_sent)
        norm_diff = torch.norm(diff)
        if norm_diff < 1e-10: return (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE)
        return diff / norm_diff

    def hypothesis_test(self, w_history):
        if len(w_history) < self.e + 1: return True
        delta   = w_history[-1] - w_history[-self.e - 1]
        matches = (torch.sign(delta) == self.s).float().sum().item()
        p       = 1.0 - binom.cdf(int(matches) - 1, self.d, 0.5)
        return p <= self.p_value

    def craft_update(self, global_flat, g_prev, w_history, n_fake):
        v_t = (torch.ones(self.d) / (self.d ** 0.5)).to(DEVICE) if self.last_g_base is None or g_prev is None else self.compute_unit_magnitude(g_prev, self.last_g_base)

        if g_prev is None:
            lambda_t = self.c_dyn.block_base
        else:
            norm_gprev = torch.norm(g_prev).item()
            if math.isnan(norm_gprev) or math.isinf(norm_gprev): norm_gprev = 1.0
            norm_gprev = min(norm_gprev, 100.0)
            succeeded  = self.hypothesis_test(w_history)
            self.c     = self.c_dyn.step(succeeded)
            lambda_t   = self.c * norm_gprev

        self.current_lambda = lambda_t

        g_backdoor = backdoor_local_train(
            global_flat, self.malicious_train_indices, self.train_dataset, self.cfg,
            epochs=self.cfg.get('backdoor_local_epochs', self.cfg['local_epochs']),
        ).to(DEVICE)

        norm_bd = torch.norm(g_backdoor)
        g_bd_normalized = (g_backdoor / norm_bd) if norm_bd > 1e-10 else torch.zeros_like(g_backdoor)

        g_sw_variance = v_t * self.s
        sw_ratio = self.cfg.get('sw_blend_ratio', 0.5)

        raw_blend = sw_ratio * g_sw_variance + (1.0 - sw_ratio) * g_bd_normalized
        blend_normalized = raw_blend / (torch.norm(raw_blend) + 1e-10)
        g_base = lambda_t * blend_normalized

        self.last_g_base = g_base.clone()

        malicious_updates = []
        for _ in range(n_fake):
            jitter = torch.randn_like(g_base) * 1e-6
            malicious_updates.append(g_base + jitter)

        return malicious_updates

# ─────────────────────────────────────────────
# 6. DEFENSES & EVALUATION
# ─────────────────────────────────────────────
def fedavg(updates, **kw):           return torch.stack(updates).mean(0)
def multi_krum(updates, n_fake=0, **kw):
    n = len(updates); f = max(0, min(n_fake, n - 1)); k = max(1, n - f - 2); m = max(1, n - f)
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j: dists.append((torch.norm(updates[i] - updates[j]).item() ** 2, j))
        dists.sort()
        scores.append((sum(d for d, _ in dists[:k]), i))
    scores.sort()
    selected = [updates[s[1]] for s in scores[:m]]
    return torch.stack(selected).mean(0)

DEFENSE_MAP = {'fedavg': fedavg, 'multi_krum': multi_krum}

def test_error_rate(model_flat, test_loader, dataset):
    model = model_from_flat(model_flat.clone(), dataset)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    return 1.0 - correct / total

# ─────────────────────────────────────────────
# 7. FL TRAINING LOOP
# ─────────────────────────────────────────────
def run_fl(cfg, output_dir):
    set_seed(cfg['seed'])
    g_rng.manual_seed(cfg['seed'])

    print(f"\n{'='*65}")
    print(f" Dataset: {cfg['dataset'].upper()} | Defense: {cfg['defense']} | Attack: {cfg['attack']}")
    backdoor_mode = cfg.get('backdoor_enable', False)
    print(f" Mode: {'SSW-PoisonedFL (BACKDOOR + SW)' if backdoor_mode else 'SW-PoisonedFL (Original)'}")
    print(f"{'='*65}")

    train_ds, test_ds = load_dataset(cfg['dataset'], data_dir=cfg.get('data_dir', './data'))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=g_rng)
    client_splits = non_iid_split(train_ds, cfg['n_genuine'], cfg['non_iid_q'])

    n_fake   = int(cfg['n_genuine'] * cfg['fake_ratio'])
    n_total  = cfg['n_genuine'] + n_fake
    n_select = max(1, int(n_total * cfg['participation']))
    
    all_clients = list(range(n_total)) 

    global_model = get_model(cfg['dataset'])
    w = flatten(global_model).to(DEVICE); d = len(w)

    attacker = None
    if cfg['attack'] == 'poisonedfl':
        if backdoor_mode:
            pool_size = cfg.get('backdoor_train_size', 300) + cfg.get('backdoor_val_size', 150) + cfg.get('backdoor_test_size', 150)
            malicious_pool = list(range(min(pool_size, len(train_ds))))
            train_end = cfg.get('backdoor_train_size', 300)
            val_end   = train_end + cfg.get('backdoor_val_size', 150)
            
            malicious_train_indices = malicious_pool[:train_end]
            malicious_val_indices   = malicious_pool[train_end:val_end]
            malicious_test_indices  = malicious_pool[val_end:]
            
            attacker = SSWPoisonedFL(d, cfg, train_ds, malicious_train_indices)
            print(f' Attack: SSW-PoisonedFL | Data Pool Sizes: Train={len(malicious_train_indices)}, Val={len(malicious_val_indices)}, Test={len(malicious_test_indices)}')
        else:
            attacker = PoisonedFL(d, cfg)
            print(' Attack: Original SW-PoisonedFL')

    w_history = deque([w.clone()], maxlen=cfg['e'] + 1)
    g_prev = None; results = []
    track_rounds = []; track_errors = []; track_bsr = []; track_clean_acc = []; track_c = []; track_lambda = []

    print(f" d={d:,} | n_genuine={cfg['n_genuine']} | n_fake={n_fake} | rounds={cfg['n_rounds']}\n")

    weights_dir = Path(output_dir) / "saved_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(1, cfg['n_rounds'] + 1), desc='Training'):
        selected_clients = random.sample(all_clients, n_select)
        genuine_selected = [c for c in selected_clients if c < cfg['n_genuine']]
        fake_selected_count = len(selected_clients) - len(genuine_selected)

        genuine_updates = [local_train(w, client_splits[i], train_ds, cfg).to(DEVICE) for i in genuine_selected]

        malicious_updates = []
        if attacker is not None and fake_selected_count > 0:
            if isinstance(attacker, SSWPoisonedFL):
                malicious_updates = attacker.craft_update(w, g_prev, w_history, fake_selected_count)
            else:
                malicious_updates = attacker.craft_update(g_prev, w_history, fake_selected_count)

        all_updates = genuine_updates + malicious_updates
        defense_fn  = DEFENSE_MAP.get(cfg['defense'], fedavg)
        
        estimated_f = max(1, int(n_select * (n_fake / n_total)))
        g_agg       = defense_fn(all_updates, n_fake=estimated_f)

        g_agg = torch.nan_to_num(g_agg, nan=0.0, posinf=100.0, neginf=-100.0)
        w = w + g_agg
        w = torch.nan_to_num(w, nan=0.0, posinf=1e4, neginf=-1e4)

        w_history.append(w.clone())
        g_prev = g_agg.clone()

        if t % 500 == 0 or (t >= 1800 and t % 50 == 0):
            attack_type = "ssw" if backdoor_mode else "sw"
            save_path = weights_dir / f"{attack_type}_model_round_{t}.pt"
            torch.save(w.detach().cpu(), save_path)

        if t % 50 == 0 or t == 1:
            if backdoor_mode:
                clean_acc, global_bsr, err = evaluate_backdoor_global(w, test_ds, cfg)
                val_bsr  = evaluate_backdoor_split(w, train_ds, malicious_val_indices, cfg)
                test_bsr = evaluate_backdoor_split(w, train_ds, malicious_test_indices, cfg)
                bsr = test_bsr 
            else:
                err       = test_error_rate(w, test_loader, cfg['dataset'])
                clean_acc = 1.0 - err
                bsr = val_bsr = test_bsr = global_bsr = 0.0

            results.append((t, err))
            curr_c      = getattr(attacker, 'c', 0.0) if attacker else 0.0
            curr_lambda = getattr(attacker, 'current_lambda', 0.0) if attacker else 0.0

            track_rounds.append(t); track_errors.append(err * 100); track_bsr.append(bsr * 100)
            track_clean_acc.append(clean_acc * 100); track_c.append(curr_c); track_lambda.append(curr_lambda)

            if backdoor_mode:
                tqdm.write(
                    f' Round {t:4d} | Error: {err*100:.1f}% | CleanAcc: {clean_acc*100:.1f}% '
                    f'| Global BSR: {global_bsr*100:.1f}% | Val BSR: {val_bsr*100:.1f}% '
                    f'| Test BSR: {test_bsr*100:.1f}% | c: {curr_c:.2f} '
                )
            else:
                tqdm.write(f' Round {t:4d} | Error: {err*100:.1f}% | c: {curr_c:.2f} | Lambda: {curr_lambda:.4f}')

    final_err = results[-1][1] if results else 0.0
    final_bsr = track_bsr[-1] if track_bsr else 0.0
    print(f'\n Final Testing Error: {final_err*100:.2f}%')
    if backdoor_mode:
        print(f' Final Malicious Test BSR: {final_bsr:.2f}%')

    tracking = {
        'rounds': track_rounds, 'errors': track_errors, 'bsr': track_bsr,
        'clean_acc': track_clean_acc, 'c': track_c, 'lambda': track_lambda,
    }
    return results, tracking

# ─────────────────────────────────────────────
# 8. PLOTTING
# ─────────────────────────────────────────────
def plot_results(tracking_sw, tracking_ssw, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SW-PoisonedFL vs SSW-PoisonedFL Comparison', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    if tracking_sw:
        ax.plot(tracking_sw['rounds'], tracking_sw['errors'], 'b--', label='SW-PoisonedFL', linewidth=2)
    if tracking_ssw:
        ax.plot(tracking_ssw['rounds'], tracking_ssw['errors'], 'r-', label='SSW-PoisonedFL', linewidth=2)
    ax.set_xlabel('Round'); ax.set_ylabel('Testing Error (%)')
    ax.set_title('Testing Error Rate vs Rounds')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if tracking_sw:
        ax.plot(tracking_sw['rounds'], tracking_sw['clean_acc'], 'b--', label='SW Clean Acc', linewidth=2)
    if tracking_ssw:
        ax.plot(tracking_ssw['rounds'], tracking_ssw['clean_acc'], 'g-', label='SSW Clean Acc', linewidth=2)
        ax.plot(tracking_ssw['rounds'], tracking_ssw['bsr'], 'r-', label='SSW Test BSR', linewidth=2)
    ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Clean Accuracy & Malicious Test BSR')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if tracking_sw:
        ax.plot(tracking_sw['rounds'], tracking_sw['c'], 'b--', label='SW c', linewidth=2)
    if tracking_ssw:
        ax.plot(tracking_ssw['rounds'], tracking_ssw['c'], 'r-', label='SSW c', linewidth=2)
    ax.set_xlabel('Round'); ax.set_ylabel('c (bounded round-to-round)')
    ax.set_title('Scaling Factor c vs Rounds')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if tracking_sw:
        ax.plot(tracking_sw['rounds'], tracking_sw['lambda'], 'b--', label='SW Lambda', linewidth=2)
    if tracking_ssw:
        ax.plot(tracking_ssw['rounds'], tracking_ssw['lambda'], 'r-', label='SSW Lambda', linewidth=2)
    ax.set_xlabel('Round'); ax.set_ylabel('Lambda')
    ax.set_title('Dynamic Lambda vs Rounds')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'ssw_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Graph saved: {plot_path}')
    plt.show()

# ─────────────────────────────────────────────
# 9. MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reproducible Federated Learning Experiment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs/plots')
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found. Please create it or pass --config path/to/config.yaml")
    
    with open(args.config, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # Setup directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")

    print('\n' + '#'*65)
    print('# EXPERIMENT 1: Original SW-PoisonedFL (No Backdoor)')
    print('#'*65)
    cfg_sw = dict(base_cfg)
    cfg_sw['backdoor_enable'] = False
    _, tracking_sw = run_fl(cfg_sw, args.output_dir)

    print('\n' + '#'*65)
    print('# EXPERIMENT 2: SSW-PoisonedFL (Backdoor + SW Combined)')
    print('#'*65)
    cfg_ssw = dict(base_cfg)
    cfg_ssw['backdoor_enable'] = True
    _, tracking_ssw = run_fl(cfg_ssw, args.output_dir)

    print('\nGenerating comparison graphs...')
    plot_results(tracking_sw, tracking_ssw, args.output_dir)
    print('Done!')

if __name__ == '__main__':
    main()


 Using device: cuda

#################################################################
# EXPERIMENT 1: Original SW-PoisonedFL (No Backdoor)
#################################################################

=================================================================
  Dataset: MNIST | Defense: multi_krum | Attack: poisonedfl
  Mode: SW-PoisonedFL (Original)
=================================================================
100%|██████████| 9.91M/9.91M [00:00<00:00, 17.8MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 477kB/s]
100%|██████████| 1.65M/1.65M [00:00<00:00, 4.47MB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 10.1MB/s]
  Attack: Original SW-PoisonedFL
  d=139,960 | n_genuine=1200 | n_fake=300 | rounds=2000

Training:   0%|          | 1/2000 [00:06<3:33:54,  6.42s/it]  Round    1 | Error: 85.9% | c: 8.50 | Lambda: 8.5000
Training:   2%|▎         | 50/2000 [02:48<2:00:50,  3.72s/it]  Round   50 | Error: 40.5% | c: 12.00 | Lambda: 0.2244
Training:   5%|▌         | 100/2000 [05:36<2:03:13,  3.89s/it]  Round  100 | Error: 17.9% | c: 11.72 | Lambda: 0.2068
Training:   8%|▊         | 150/2000 [08:30<2:09:19,  4.19s/it]  Round  150 | Error: 13.7% | c: 12.00 | Lambda: 0.2105
Training:  10%|█         | 200/2000 [11:18<1:52:49,  3.76s/it]  Round  200 | Error: 11.9% | c: 11.70 | Lambda: 0.1708
Training:  12%|█▎        | 250/2000 [14:06<1:57:01,  4.01s/it]  Round  250 | Error: 11.2% | c: 12.00 | Lambda: 0.2416
Training:  15%|█▌        | 300/2000 [16:49<1:48:13,  3.82s/it]  Round  300 | Error: 10.5% | c: 12.00 | Lambda: 0.1966
Training:  18%|█▊        | 350/2000 [19:37<1:43:13,  3.75s/it]  Round  350 | Error: 10.0% | c: 11.69 | Lambda: 0.2046
Training:  20%|██        | 400/2000 [22:18<1:39:51,  3.74s/it]  Round  400 | Error: 9.6% | c: 12.00 | Lambda: 0.2244
Training:  22%|██▎       | 450/2000 [25:04<1:39:40,  3.86s/it]  Round  450 | Error: 9.9% | c: 11.65 | Lambda: 0.1736
Training:  25%|██▌       | 500/2000 [27:50<1:36:27,  3.86s/it]  Round  500 | Error: 9.7% | c: 12.00 | Lambda: 0.6588
Training:  28%|██▊       | 550/2000 [30:37<1:41:59,  4.22s/it]  Round  550 | Error: 9.3% | c: 12.00 | Lambda: 0.4563
Training:  30%|███       | 600/2000 [33:23<1:28:34,  3.80s/it]  Round  600 | Error: 9.9% | c: 12.00 | Lambda: 0.2708
Training:  32%|███▎      | 650/2000 [36:13<1:25:46,  3.81s/it]  Round  650 | Error: 9.2% | c: 11.90 | Lambda: 0.6574
Training:  35%|███▌      | 700/2000 [39:05<1:36:00,  4.43s/it]  Round  700 | Error: 10.0% | c: 12.00 | Lambda: 0.3486
Training:  38%|███▊      | 750/2000 [41:53<1:16:31,  3.67s/it]  Round  750 | Error: 12.3% | c: 11.82 | Lambda: 0.4631
Training:  40%|████      | 800/2000 [44:42<1:20:10,  4.01s/it]  Round  800 | Error: 13.3% | c: 12.00 | Lambda: 0.8974
Training:  42%|████▎     | 850/2000 [47:30<1:17:29,  4.04s/it]  Round  850 | Error: 14.3% | c: 11.93 | Lambda: 0.4873
Training:  45%|████▌     | 900/2000 [50:22<1:11:13,  3.88s/it]  Round  900 | Error: 14.2% | c: 11.95 | Lambda: 0.5115
Training:  48%|████▊     | 950/2000 [53:10<1:14:49,  4.28s/it]  Round  950 | Error: 14.0% | c: 12.00 | Lambda: 0.8333
Training:  50%|█████     | 1000/2000 [55:58<1:01:45,  3.71s/it]  Round 1000 | Error: 15.2% | c: 12.00 | Lambda: 0.5746
Training:  52%|█████▎    | 1050/2000 [58:45<59:57,  3.79s/it]  Round 1050 | Error: 17.8% | c: 11.90 | Lambda: 1.2427
Training:  55%|█████▌    | 1100/2000 [1:01:27<55:51,  3.72s/it]  Round 1100 | Error: 19.5% | c: 11.73 | Lambda: 1.3355
Training:  57%|█████▊    | 1150/2000 [1:04:08<52:15,  3.69s/it]  Round 1150 | Error: 21.3% | c: 12.00 | Lambda: 2.0892
Training:  60%|██████    | 1200/2000 [1:06:50<48:46,  3.66s/it]  Round 1200 | Error: 28.3% | c: 11.75 | Lambda: 0.8540
Training:  62%|██████▎   | 1250/2000 [1:09:33<52:47,  4.22s/it]  Round 1250 | Error: 36.2% | c: 12.00 | Lambda: 0.8931
Training:  65%|██████▌   | 1300/2000 [1:12:22<44:25,  3.81s/it]  Round 1300 | Error: 42.9% | c: 11.80 | Lambda: 1.3427
Training:  68%|██████▊   | 1350/2000 [1:15:04<39:03,  3.60s/it]  Round 1350 | Error: 47.8% | c: 11.82 | Lambda: 0.8387
Training:  70%|███████   | 1400/2000 [1:17:52<43:46,  4.38s/it]  Round 1400 | Error: 54.9% | c: 12.00 | Lambda: 1.3755
Training:  72%|███████▎  | 1450/2000 [1:20:39<34:53,  3.81s/it]  Round 1450 | Error: 68.2% | c: 12.00 | Lambda: 1.5219
Training:  75%|███████▌  | 1500/2000 [1:23:22<35:36,  4.27s/it]  Round 1500 | Error: 72.1% | c: 11.86 | Lambda: 5.5694
Training:  78%|███████▊  | 1550/2000 [1:26:05<27:14,  3.63s/it]  Round 1550 | Error: 76.8% | c: 12.00 | Lambda: 2.3529
Training:  80%|████████  | 1600/2000 [1:28:48<27:47,  4.17s/it]  Round 1600 | Error: 90.8% | c: 11.96 | Lambda: 0.0000
Training:  82%|████████▎ | 1650/2000 [1:31:31<21:48,  3.74s/it]  Round 1650 | Error: 90.8% | c: 8.97 | Lambda: 0.0000
Training:  85%|████████▌ | 1700/2000 [1:34:17<20:00,  4.00s/it]  Round 1700 | Error: 90.8% | c: 7.39 | Lambda: 0.0000
Training:  88%|████████▊ | 1750/2000 [1:37:05<16:43,  4.01s/it]  Round 1750 | Error: 90.8% | c: 7.00 | Lambda: 0.0000
Training:  90%|█████████ | 1800/2000 [1:39:45<12:30,  3.75s/it]  Round 1800 | Error: 90.8% | c: 7.10 | Lambda: 0.0000
Training:  92%|█████████▎| 1850/2000 [1:42:27<09:24,  3.77s/it]  Round 1850 | Error: 90.8% | c: 7.00 | Lambda: 0.0000
Training:  95%|█████████▌| 1900/2000 [1:45:07<06:40,  4.01s/it]  Round 1900 | Error: 90.8% | c: 7.13 | Lambda: 0.0000
Training:  98%|█████████▊| 1950/2000 [1:47:50<03:10,  3.81s/it]  Round 1950 | Error: 90.8% | c: 7.43 | Lambda: 0.0000
Training: 100%|██████████| 2000/2000 [1:50:34<00:00,  3.32s/it]
  Round 2000 | Error: 90.8% | c: 7.00 | Lambda: 0.0000

  Final Testing Error: 90.76%

#################################################################
# EXPERIMENT 2: SSW-PoisonedFL (Backdoor + SW Combined)
#################################################################

=================================================================
  Dataset: MNIST | Defense: multi_krum | Attack: poisonedfl
  Mode: SSW-PoisonedFL (BACKDOOR + SW)
=================================================================
  Attack: SSW-PoisonedFL | Data Pool Sizes: Train=300, Val=150, Test=150
  d=139,960 | n_genuine=1200 | n_fake=300 | rounds=2000

Training:   0%|          | 1/2000 [00:05<3:15:41,  5.87s/it]  Round    1 | Error: 90.2% | CleanAcc: 9.8% | Global BSR: 100.0% | Val BSR: 100.0% | Test BSR: 100.0% | c: 8.50 
Training:   2%|▎         | 50/2000 [02:59<2:11:33,  4.05s/it]  Round   50 | Error: 34.5% | CleanAcc: 65.5% | Global BSR: 11.8% | Val BSR: 6.4% | Test BSR: 10.4% | c: 12.00 
Training:   5%|▌         | 100/2000 [06:02<2:13:26,  4.21s/it]  Round  100 | Error: 15.7% | CleanAcc: 84.3% | Global BSR: 1.4% | Val BSR: 0.7% | Test BSR: 0.7% | c: 11.72 
Training:   8%|▊         | 150/2000 [09:07<2:16:52,  4.44s/it]  Round  150 | Error: 13.8% | CleanAcc: 86.2% | Global BSR: 4.4% | Val BSR: 2.1% | Test BSR: 3.0% | c: 12.00 
Training:  10%|█         | 200/2000 [12:10<2:05:27,  4.18s/it]  Round  200 | Error: 11.6% | CleanAcc: 88.4% | Global BSR: 2.8% | Val BSR: 1.4% | Test BSR: 2.2% | c: 11.70 
Training:  12%|█▎        | 250/2000 [15:12<2:03:12,  4.22s/it]  Round  250 | Error: 9.8% | CleanAcc: 90.2% | Global BSR: 0.9% | Val BSR: 0.0% | Test BSR: 0.7% | c: 12.00 
Training:  15%|█▌        | 300/2000 [18:09<1:52:39,  3.98s/it]  Round  300 | Error: 8.9% | CleanAcc: 91.1% | Global BSR: 0.8% | Val BSR: 0.0% | Test BSR: 0.7% | c: 12.00 
Training:  18%|█▊        | 350/2000 [21:08<1:53:55,  4.14s/it]  Round  350 | Error: 8.3% | CleanAcc: 91.7% | Global BSR: 0.8% | Val BSR: 0.0% | Test BSR: 0.7% | c: 11.69 
Training:  20%|██        | 400/2000 [24:02<1:50:23,  4.14s/it]  Round  400 | Error: 7.7% | CleanAcc: 92.3% | Global BSR: 1.3% | Val BSR: 0.0% | Test BSR: 0.7% | c: 12.00 
Training:  22%|██▎       | 450/2000 [26:59<1:43:20,  4.00s/it]  Round  450 | Error: 7.2% | CleanAcc: 92.8% | Global BSR: 1.6% | Val BSR: 0.0% | Test BSR: 1.5% | c: 11.65 
Training:  25%|██▌       | 500/2000 [29:59<1:46:17,  4.25s/it]  Round  500 | Error: 6.7% | CleanAcc: 93.3% | Global BSR: 1.0% | Val BSR: 0.0% | Test BSR: 0.7% | c: 12.00 
Training:  28%|██▊       | 550/2000 [32:57<1:43:05,  4.27s/it]  Round  550 | Error: 6.3% | CleanAcc: 93.7% | Global BSR: 2.6% | Val BSR: 0.7% | Test BSR: 4.4% | c: 12.00 
Training:  30%|███       | 600/2000 [35:56<1:35:26,  4.09s/it]  Round  600 | Error: 6.0% | CleanAcc: 94.0% | Global BSR: 2.9% | Val BSR: 1.4% | Test BSR: 4.4% | c: 12.00 
Training:  32%|███▎      | 650/2000 [38:56<1:29:58,  4.00s/it]  Round  650 | Error: 5.6% | CleanAcc: 94.4% | Global BSR: 4.8% | Val BSR: 3.5% | Test BSR: 5.9% | c: 11.90 
Training:  35%|███▌      | 700/2000 [41:58<1:36:41,  4.46s/it]  Round  700 | Error: 5.2% | CleanAcc: 94.8% | Global BSR: 6.3% | Val BSR: 4.3% | Test BSR: 6.7% | c: 12.00 
Training:  38%|███▊      | 750/2000 [44:58<1:24:42,  4.07s/it]  Round  750 | Error: 4.9% | CleanAcc: 95.1% | Global BSR: 5.8% | Val BSR: 3.5% | Test BSR: 5.2% | c: 11.82 
Training:  40%|████      | 800/2000 [47:55<1:22:26,  4.12s/it]  Round  800 | Error: 4.7% | CleanAcc: 95.3% | Global BSR: 6.3% | Val BSR: 5.0% | Test BSR: 5.2% | c: 12.00 
Training:  42%|████▎     | 850/2000 [50:59<1:26:54,  4.53s/it]  Round  850 | Error: 4.5% | CleanAcc: 95.5% | Global BSR: 14.9% | Val BSR: 14.9% | Test BSR: 17.0% | c: 11.93 
Training:  45%|████▌     | 900/2000 [54:03<1:17:24,  4.22s/it]  Round  900 | Error: 4.4% | CleanAcc: 95.6% | Global BSR: 32.7% | Val BSR: 36.9% | Test BSR: 42.2% | c: 11.95 
Training:  48%|████▊     | 950/2000 [57:05<1:18:00,  4.46s/it]  Round  950 | Error: 4.2% | CleanAcc: 95.8% | Global BSR: 34.8% | Val BSR: 39.7% | Test BSR: 44.4% | c: 12.00 
Training:  50%|█████     | 1000/2000 [1:00:09<1:07:46,  4.07s/it]  Round 1000 | Error: 4.2% | CleanAcc: 95.8% | Global BSR: 43.9% | Val BSR: 48.9% | Test BSR: 53.3% | c: 12.00 
Training:  52%|█████▎    | 1050/2000 [1:03:16<1:05:22,  4.13s/it]  Round 1050 | Error: 4.0% | CleanAcc: 96.0% | Global BSR: 53.9% | Val BSR: 59.6% | Test BSR: 61.5% | c: 11.90 
Training:  55%|█████▌    | 1100/2000 [1:06:16<1:03:23,  4.23s/it]  Round 1100 | Error: 4.0% | CleanAcc: 96.0% | Global BSR: 42.6% | Val BSR: 49.6% | Test BSR: 47.4% | c: 11.73 
Training:  57%|█████▊    | 1150/2000 [1:09:14<56:21,  3.98s/it]  Round 1150 | Error: 4.2% | CleanAcc: 95.8% | Global BSR: 72.6% | Val BSR: 83.0% | Test BSR: 82.2% | c: 12.00 
Training:  60%|██████    | 1200/2000 [1:12:15<55:08,  4.14s/it]  Round 1200 | Error: 4.1% | CleanAcc: 95.9% | Global BSR: 79.0% | Val BSR: 84.4% | Test BSR: 84.4% | c: 11.75 
Training:  62%|██████▎   | 1250/2000 [1:15:16<58:06,  4.65s/it]  Round 1250 | Error: 4.1% | CleanAcc: 95.9% | Global BSR: 81.3% | Val BSR: 85.8% | Test BSR: 86.7% | c: 12.00 
Training:  65%|██████▌   | 1300/2000 [1:18:19<46:35,  3.99s/it]  Round 1300 | Error: 5.5% | CleanAcc: 94.5% | Global BSR: 96.6% | Val BSR: 97.9% | Test BSR: 97.8% | c: 11.80 
Training:  68%|██████▊   | 1350/2000 [1:21:17<43:48,  4.04s/it]  Round 1350 | Error: 5.0% | CleanAcc: 95.0% | Global BSR: 97.7% | Val BSR: 98.6% | Test BSR: 98.5% | c: 11.82 
Training:  70%|███████   | 1400/2000 [1:24:20<46:39,  4.67s/it]  Round 1400 | Error: 6.0% | CleanAcc: 94.0% | Global BSR: 99.3% | Val BSR: 98.6% | Test BSR: 100.0% | c: 12.00 
Training:  72%|███████▎  | 1450/2000 [1:27:22<37:01,  4.04s/it]  Round 1450 | Error: 5.3% | CleanAcc: 94.7% | Global BSR: 99.3% | Val BSR: 98.6% | Test BSR: 99.3% | c: 12.00 
Training:  75%|███████▌  | 1500/2000 [1:30:19<39:01,  4.68s/it]  Round 1500 | Error: 5.3% | CleanAcc: 94.7% | Global BSR: 99.6% | Val BSR: 99.3% | Test BSR: 100.0% | c: 11.86 
Training:  78%|███████▊  | 1550/2000 [1:33:17<29:47,  3.97s/it]  Round 1550 | Error: 5.0% | CleanAcc: 95.0% | Global BSR: 99.5% | Val BSR: 98.6% | Test BSR: 100.0% | c: 12.00 
Training:  80%|████████  | 1600/2000 [1:36:16<29:40,  4.45s/it]  Round 1600 | Error: 5.7% | CleanAcc: 94.3% | Global BSR: 99.0% | Val BSR: 98.6% | Test BSR: 99.3% | c: 11.96 
Training:  82%|████████▎ | 1650/2000 [1:39:15<23:42,  4.06s/it]  Round 1650 | Error: 7.7% | CleanAcc: 92.3% | Global BSR: 98.8% | Val BSR: 97.9% | Test BSR: 99.3% | c: 11.89 
Training:  85%|████████▌ | 1700/2000 [1:42:18<21:31,  4.31s/it]  Round 1700 | Error: 8.6% | CleanAcc: 91.4% | Global BSR: 97.6% | Val BSR: 97.9% | Test BSR: 99.3% | c: 12.00 
Training:  88%|████████▊ | 1750/2000 [1:45:20<17:35,  4.22s/it]  Round 1750 | Error: 9.7% | CleanAcc: 90.3% | Global BSR: 97.5% | Val BSR: 97.2% | Test BSR: 98.5% | c: 11.97 
Training:  90%|█████████ | 1800/2000 [1:48:14<13:35,  4.08s/it]  Round 1800 | Error: 14.5% | CleanAcc: 85.5% | Global BSR: 99.5% | Val BSR: 99.3% | Test BSR: 99.3% | c: 12.00 
Training:  92%|█████████▎| 1850/2000 [1:51:13<10:00,  4.00s/it]  Round 1850 | Error: 9.7% | CleanAcc: 90.3% | Global BSR: 96.1% | Val BSR: 96.5% | Test BSR: 97.8% | c: 11.76 
Training:  95%|█████████▌| 1900/2000 [1:54:08<07:04,  4.25s/it]  Round 1900 | Error: 10.5% | CleanAcc: 89.5% | Global BSR: 95.1% | Val BSR: 95.0% | Test BSR: 97.0% | c: 12.00 
Training:  98%|█████████▊| 1950/2000 [1:57:10<03:26,  4.13s/it]  Round 1950 | Error: 9.7% | CleanAcc: 90.3% | Global BSR: 94.0% | Val BSR: 95.7% | Test BSR: 97.0% | c: 12.00 
Training: 100%|██████████| 2000/2000 [2:00:10<00:00,  3.61s/it]
  Round 2000 | Error: 12.9% | CleanAcc: 87.1% | Global BSR: 97.5% | Val BSR: 97.2% | Test BSR: 97.8% | c: 11.75 

  Final Testing Error: 12.87%
  Final Malicious Test BSR: 97.78%

Generating  comparison graphs...

Graph saved: ssw_results.png
Done!

