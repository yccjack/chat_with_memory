import urllib.request
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=2):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)

    return dataloader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + stride:i + max_length + stride]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


if __name__ == "__main__":
    print("tiktoken version:", version("tiktoken"))
    print("torch version:", version("torch"))
    print("torch cuda:", torch.cuda.is_available())

    file_path = "the-verdict.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_txt = f.read()
    dataloader = create_dataloader_v1(raw_txt, batch_size=1, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(inputs)
    print(targets)
