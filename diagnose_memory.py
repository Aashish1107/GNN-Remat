"""
Diagnose why GNN-Remat uses more memory than baseline for GCN/SAGE.
Measures live memory after forward pass to see what the autograd graph retains.
"""
import torch
import copy
from gnn_remat import gnn_remat
from gnn_remat.benchmark.models import build

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
n, e, f = 5000, 50000, 128
x  = torch.randn(n, f, device=device)
ei = torch.stack([torch.randint(0, n, (e,), device=device),
                  torch.randint(0, n, (e,), device=device)])

def profile_model(label, model):
    model = model.to(device).train()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # warmup
    for _ in range(2):
        opt.zero_grad(); model(x, ei).sum().backward(); opt.step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt.zero_grad()
    out = model(x, ei)
    live_after_fwd = torch.cuda.memory_allocated() / 1e6
    peak_after_fwd = torch.cuda.max_memory_allocated() / 1e6
    out.sum().backward()
    peak_after_bwd = torch.cuda.max_memory_allocated() / 1e6
    opt.step()
    print(f"  {label:<30}  live_after_fwd={live_after_fwd:7.1f} MB"
          f"  peak_fwd={peak_after_fwd:7.1f} MB  peak_bwd={peak_after_bwd:7.1f} MB")

for mname in ["gcn", "graphsage", "gat"]:
    kw = dict(in_channels=f, hidden=256, out_channels=40, num_layers=3)
    if mname == "gat":
        kw["heads"] = 4
    base  = build(mname, **kw)
    remat = gnn_remat(copy.deepcopy(base))
    print(f"\n=== {mname.upper()} ===")
    profile_model("baseline", base)
    profile_model("gnn_remat (propagate)", remat)
