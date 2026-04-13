# debug.py  — new file, run this
import torch
from transformer.dataset import DataBase, pickleDataset

base_data = DataBase("DATASET/umls", subgraph="DATASET/umls/subgraph3")

class Opt: pass
opt = Opt()
opt.data="DATASET/umls"; opt.jump=3; opt.padding=140
opt.n_head=4; opt.d_v=32; opt.n_layers=6; opt.dropout=0.1
opt.d_k=opt.d_v; opt.d_model=opt.n_head*opt.d_v
opt.d_word_vec=opt.d_model; opt.d_inner_hid=opt.d_model*4
opt.subgraph=opt.data+f'/subgraph{opt.jump}'
opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()
opt.src_pad_idx = base_data.e2id['<pad>']

id2e = {v: k for k, v in base_data.e2id.items()}

test_data = pickleDataset(base_data, opt, mode='test')
loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False,
    collate_fn=pickleDataset.collate_fn
)

for batch in loader:
    subgraph, link, target, tailIndexs, length = batch
    print("subgraph shape:", subgraph.shape)
    print("link shape    :", link.shape)
    print("target shape  :", target.shape)
    print("tailIndexs    :", tailIndexs.shape)
    print("length        :", length.shape)
    print()
    print("subgraph[0,:5]:", subgraph[0, :5].tolist())
    print("target[0,:5]  :", target[0, :5].tolist())
    print()
    print("subgraph[0,0] as entity  :", id2e.get(subgraph[0,0].item()))
    print("target[0,0]   as entity  :", id2e.get(target[0,0].item()))
    print("target[0,1]   as entity  :", id2e.get(target[0,1].item()))
    print("target[0,1]   as relation:", base_data.id2r.get(target[0,1].item()))
    break