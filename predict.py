import torch
import torch.nn.functional as F
from train import load_model
from transformer.dataset import DataBase, pickleDataset
from transformer.Translator import Translator

class Opt: pass
opt = Opt()
opt.data        = "DATASET/umls"
opt.jump        = 3
opt.padding     = 140
opt.n_head      = 4
opt.d_v         = 32
opt.n_layers    = 6
opt.dropout     = 0.1
opt.d_k         = opt.d_v
opt.d_model     = opt.n_head * opt.d_v
opt.d_word_vec  = opt.d_model
opt.d_inner_hid = opt.d_model * 4
opt.subgraph    = opt.data + f'/subgraph{opt.jump}'
opt.decode_rule = False
opt.the_rel     = 0.6
opt.the_rel_min = 0.3
opt.the_all     = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_data = DataBase(opt.data, subgraph=opt.subgraph)
opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()
opt.src_pad_idx = base_data.e2id['<pad>']

translator = Translator(
    model     = load_model(opt, device, base_data.nebor_relation),
    opt       = opt,
    device    = device,
    base_data = base_data,
).to(device)

translator.load_state_dict(
    torch.load("EXPS/umls-distilbert/model_epoch_3.pt", map_location=device),
    strict=False
)
translator.eval()
print("Model loaded.\n")

# ---- lookup maps ----
id2e = {v: k for k, v in base_data.e2id.items()}
r2id = {v: k for k, v in base_data.id2r.items()}

# ---- build subgraph on the fly from any (head, relation) ----
def build_subgraph(head_id, rel_id, padding=140, pad_idx=0):
    """
    Walk the KG neighbor graph up to `jump` hops from head_id
    and return a padded sequence of entity ids.
    """
    visited = [head_id]
    frontier = [head_id]

    for _ in range(opt.jump):
        next_frontier = []
        for eid in frontier:
            if eid in base_data.neighbors:
                for r, tails in base_data.neighbors[eid].items():
                    for t in tails:
                        if t not in visited:
                            visited.append(t)
                            next_frontier.append(t)
                            if len(visited) >= padding:
                                break
                    if len(visited) >= padding:
                        break
            if len(visited) >= padding:
                break
        frontier = next_frontier
        if not frontier:
            break

    # pad / truncate to exactly `padding` length
    seq = visited[:padding]
    seq += [pad_idx] * (padding - len(seq))

    subgraph  = torch.tensor([seq], dtype=torch.long)          # [1, padding]

    # target: [head, relation, 0(unknown tail)]
    target    = torch.tensor([[head_id, rel_id, 0]], dtype=torch.long)  # [1, 3]

    # tailIndexs: placeholder (not used in 'test' scoring without true tail)
    tailIndexs = torch.zeros(1, 1, dtype=torch.long)

    # length: [1, jump+2] placeholder
    length    = torch.zeros(1, opt.jump + 2, dtype=torch.long)

    # link: build neighbor relation matrix (same as dataset does internally)
    # use the precomputed nebor_relation from base_data
    link = base_data.nebor_relation.unsqueeze(0)  # [1, ...]

    return subgraph, link, target, tailIndexs, length


def predict(head_name, relation_name, topk=5):
    head_id     = base_data.e2id.get(head_name)
    relation_id = r2id.get(relation_name)

    if head_id is None:
        print(f"  Entity '{head_name}' not found.")
        return
    if relation_id is None:
        print(f"  Relation '{relation_name}' not found.")
        return

    subgraph, link, target, tailIndexs, length = build_subgraph(head_id, relation_id)

    with torch.no_grad():
        # run raw model to get attention over all entities
        enc_out, = translator.model.encoder(
            subgraph.to(device),
            (subgraph != opt.src_pad_idx).long().unsqueeze(1).to(device)
        )
        init_seq = target[:, 1].unsqueeze(-1).to(device)   # [1,1] relation token
        dec_out  = translator._model_decode(init_seq, enc_out,
                    (subgraph != opt.src_pad_idx).long().unsqueeze(1).to(device))

        # forwardAllNLP-style: propagate through graph using attention weights
        # instead we just take softmax over the decoder output as entity scores
        scores = F.softmax(dec_out[:, -1, :], dim=-1)   # [1, trg_vocab]

        topk_scores, topk_ids = scores.topk(topk)

    print(f"\n  ===== TOP {topk} PREDICTIONS =====")
    print(f"  Head     : {head_name}")
    print(f"  Relation : {relation_name}")
    print(f"  {'Rank':<6} {'Entity':<35} {'Score'}")
    print(f"  {'-'*55}")
    for rank, (score, eid) in enumerate(zip(topk_scores[0], topk_ids[0]), 1):
        entity = id2e.get(eid.item(), f"id={eid.item()}")
        print(f"  {rank:<6} {entity:<35} {score.item():.4f}")
    print()


# ---- show all entities and relations ----
entities  = sorted([k for k in base_data.e2id.keys() if k != '<pad>'])
relations = sorted(list(base_data.id2r.values()))

print("=" * 55)
print(f"Total entities : {len(entities)}")
print(f"Total relations: {len(relations)}")
print()
print("Sample entities :", entities[:6])
print("Sample relations:", relations[:6])
print("=" * 55)

# ---- interactive loop ----
while True:
    print()
    head_name = input("Enter head entity  (or 'q' to quit): ").strip()
    if head_name.lower() == 'q':
        break
    relation_name = input("Enter relation     (or 'q' to quit): ").strip()
    if relation_name.lower() == 'q':
        break
    predict(head_name, relation_name, topk=5)