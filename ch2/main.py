import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self,tokenizer,txt,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0,len(token_ids)-max_length,stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self): # len(dataset)にアクセスするため。
        return len(self.input_ids)

    def __getitem__(self, idx): # dataset[idx]を取得するため。
        return self.input_ids[idx],self.target_ids[idx]

# dataloaderを作成する。
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(tokenizer,txt,max_length,stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text,batch_size=2,max_length=4,stride=2,shuffle=False)
data_iter = iter(dataloader) # DataLoaderをイテレータに変換。
first_batch = next(data_iter) # イテレータから次のバッチを取得。 
second_batch = next(data_iter)
print(first_batch)
print(second_batch)