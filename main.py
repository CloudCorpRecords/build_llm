# main.py

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2LMHeadModel, get_linear_schedule_with_warmup, GPT2TokenizerFast
from torchtext.datasets import AG_NEWS

# specify the model name
model_name = "distilgpt2"

# load the tokenizer and the model
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# add this line to ensure pad token exists in tokenizer
tokenizer.pad_token = tokenizer.eos_token

# specify your dataset
train_iter, test_iter = AG_NEWS(root='.')
lines = [torch.tensor(tokenizer.encode(str(x[1]), truncation=True, max_length=512)) for x in train_iter]

# create a collate function to pad sequences in a batch
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

# create a DataLoader to handle batching of inputs
batch_size = 16
dataloader = DataLoader(lines, batch_size=batch_size, collate_fn=collate_fn)

# specify the learning rate
lr = 1e-5

# set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# set up the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=-1)

# train the model
for epoch in range(5):  # number of epochs
    for idx, batch in enumerate(dataloader):
        inputs = batch
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if idx % 100 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

    print(f"Epoch {epoch+1} completed")

# save the model
model.save_pretrained("./saved_model")

