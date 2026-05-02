"""Tests for core/detector.py"""
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from gnn_remat.core.detector import detect, filter_by_name, filter_by_type


class FlatGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 4)
    def forward(self, x, ei): return self.conv2(self.conv1(x, ei).relu(), ei)

class NestedGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FlatGCN()
    def forward(self, x, ei): return self.encoder(x, ei)

class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = GCNConv(8, 16)
        self.conv2  = SAGEConv(16, 4)
        self.linear = nn.Linear(4, 2)
    def forward(self, x, ei): return self.linear(self.conv2(self.conv1(x, ei).relu(), ei).relu())


# detect()
def test_detect_finds_two_layers():        
    assert len(detect(FlatGCN())) == 2
def test_detect_names_in_order():          
    assert [i.name for i in detect(FlatGCN())] == ["conv1", "conv2"]
def test_detect_recurses_into_nested():    
    assert [i.name for i in detect(NestedGCN())] == ["encoder.conv1", "encoder.conv2"]
def test_detect_skips_linear():            
    assert "linear" not in [i.name for i in detect(MixedModel())]
def test_detect_empty_model():             
    assert detect(nn.Sequential(nn.Linear(4,4))) == []

def test_detect_parent_attr_correct():
    model = FlatGCN()
    info  = detect(model)[0]
    assert info.parent is model
    assert info.attr   == "conv1"
    assert getattr(info.parent, info.attr) is model.conv1

# filter_by_name()
def test_filter_by_name_keeps_matching():  
    assert len(filter_by_name(detect(FlatGCN()), ["conv1"])) == 1
def test_filter_by_name_empty_list():      
    assert filter_by_name(detect(FlatGCN()), []) == []
def test_filter_by_name_unknown():         
    assert filter_by_name(detect(FlatGCN()), ["xyz"]) == []

# filter_by_type()
def test_filter_by_type_gcnconv():
    filtered = filter_by_type(detect(MixedModel()), [GCNConv])
    assert len(filtered) == 1 and isinstance(filtered[0].module, GCNConv)

def test_filter_by_type_multiple():        
    assert len(filter_by_type(detect(MixedModel()), [GCNConv, SAGEConv])) == 2
def test_filter_by_type_no_match():        
    assert filter_by_type(detect(FlatGCN()), [GATConv]) == []
