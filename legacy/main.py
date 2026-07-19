import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class GrokData(Dataset):
    def __init__(self, file):
        self.data = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                self.data.append(list(map(int, line.split())))
        self.data = torch.tensor(self.data, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

class CausalTrans(nn.Module):
    def __init__(self, vocabSz=97, seqLen=2, dModel=128, nHead=4, nLyr=2):
        super().__init__()
        self.tokEmb = nn.Embedding(vocabSz, dModel)
        self.posEmb = nn.Embedding(seqLen, dModel)
        lyr = nn.TransformerEncoderLayer(dModel, nHead, dModel*4, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(lyr, nLyr, enable_nested_tensor=False)
        self.out = nn.Linear(dModel, vocabSz)
        mask = torch.triu(torch.ones(seqLen, seqLen)*float('-inf'), diagonal=1)
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        b, l = x.shape
        pos = torch.arange(l, device=x.device)
        emb = self.tokEmb(x) + self.posEmb(pos)
        h = self.trans(emb, mask=self.mask[:l,:l], is_causal=True)
        return self.out(h)

def trainRun():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainDs = GrokData('data/train_grok.txt')
    valDs = GrokData('data/test_grok.txt')
    
    train_data_gpu = trainDs.data.to(device)
    val_data_gpu = valDs.data.to(device)
    train_x, train_y = train_data_gpu[:, :2], train_data_gpu[:, 2]
    val_x, val_y = val_data_gpu[:, :2], val_data_gpu[:, 2]
    model = CausalTrans().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5)
    writer = SummaryWriter('runs/grokking_mod97_congruence')
    P = 97
    k_tensor = torch.arange(P, device=device).unsqueeze(0)
    epochs = 5000
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(train_x)
            logits = pred[:, -1, :]
            ce_loss = F.cross_entropy(logits, train_y)
            probs = F.softmax(logits, dim=-1)
            target = train_y.unsqueeze(1)
            cos_penalty = 1.0 - torch.cos(2 * torch.pi * (k_tensor - target) / P)
            chk_loss = torch.sum(probs * cos_penalty, dim=-1).mean()
            loss = ce_loss + 0.5 * chk_loss
        loss.backward()
        opt.step()
        train_acc = (logits.argmax(-1) == train_y).float().mean().item()
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    val_pred = model(val_x)[:, -1, :]
                    val_ce_loss = F.cross_entropy(val_pred, val_y).item()
                    val_acc = (val_pred.argmax(-1) == val_y).float().mean().item()
                wNorm = sum(p.norm(2).item()**2 for p in model.parameters())**0.5
            writer.add_scalar('Loss/Train_CE', ce_loss.item(), ep)
            writer.add_scalar('Loss/Train_Congruence', chk_loss.item(), ep)
            writer.add_scalar('Loss/Val_CE', val_ce_loss, ep)
            writer.add_scalar('Acc/Train', train_acc, ep)
            writer.add_scalar('Acc/Val', val_acc, ep)
            writer.add_scalar('Dynamics/WeightNorm', wNorm, ep)
            print(f"Ep:{ep:05d} | TrLoss:{ce_loss.item():.3f} | ValLoss:{val_ce_loss:.3f} | TrAcc:{train_acc*100:.1f}% | ValAcc:{val_acc*100:.1f}% | WNorm:{wNorm:.1f}")
    writer.close()

if __name__ == '__main__':
    trainRun()

