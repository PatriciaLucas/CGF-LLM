from AUTODCETS import util, feature_selection
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import GPT2Model, GPT2Config

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import clshq_tk
from clshq_tk.modules.fuzzy import GridPartitioner, trimf, gaussmf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE, DEFAULT_PATH, resume, checkpoint, order_window


def causal_text(df, name_dataset, target, max_lags, tokenizer):

    variables = df.columns.tolist()

    # Causal graph generation
    scaler = {}
    labels_scaled = {}
    inputs = {}
    X_list, y_list = [], []
    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)
    arr = True
    for v in variables:
      X, y_hat = util.organize_dataset(df, graph[v], max_lags, v)
      X_list.append(X)
      y = df[v][1].squeeze().tolist()[max_lags:]
      y = np.asarray(y)
      y_list.append(y)

      scaler[v] = StandardScaler()
      if arr:
        labels_scaled = scaler[v].fit_transform(y.reshape(-1, 1)).copy()
      else:
        labels_scaled = np.hstack((labels_scaled, scaler[v].fit_transform(y.reshape(-1, 1))))
      arr = False

    inputs = fcllm.create_sequences_input(X, y, tokenizer)

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(
    inputs,
    padding_side='left',
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
    )

    #input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")

    return fcllm.custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler, tokenizer, inputs

def text(df, name_dataset, target, max_lags, tokenizer):
    variables = df.columns.tolist()
    scaler = {}
    labels_scaled = {}
    inputs = {}
    X_list, y_list = [], []
    # Complete graph generation
    graph = feature_selection.complete_graph(df.head(2000), target=target, max_lags=max_lags)
    arr = True
    for v in variables:
      X, y_hat = util.organize_dataset(df, graph[v], max_lags, v)
      X_list.append(X)
      y = df[v][1].squeeze().tolist()[max_lags:]
      y = np.asarray(y)
      y_list.append(y)

      scaler[v] = StandardScaler()
      if arr:
        labels_scaled = scaler[v].fit_transform(y.reshape(-1, 1)).copy()
      else:
        labels_scaled = np.hstack((labels_scaled, scaler[v].fit_transform(y.reshape(-1, 1))))
      arr = False

    inputs = fcllm.create_sequences_input(X_list, y_list, tokenizer)

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    #input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")
    input_tokens = tokenizer(
    inputs,
    padding_side='left',
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
    )

    return fcllm.custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler, tokenizer, inputs


def fuzzy_causal(df, name_dataset, target, max_lags, tokenizer, partitions):

    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)

    # Fuzzification of time series
    data_fuzzy = pd.DataFrame(columns=variables)
    for v in variables:
        dict_variables[v] = fcllm.fuzzification(pd.DataFrame(df[v]), name_dataset, v, partitions, None)
        data_fuzzy[v] = dict_variables[v][0]

    # Causal graph generation
    scaler = {}
    labels_scaled = {}
    inputs = {}
    X_list, y_list = [], []
    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)
    arr = True
    for v in variables:
      X, y_hat = util.organize_dataset(data_fuzzy, graph[v], max_lags, v)
      X_list.append(X)
      y = dict_variables[v][1].squeeze().tolist()[max_lags:]
      y = np.asarray(y)
      y_list.append(y)

      scaler[v] = StandardScaler()
      if arr:
        labels_scaled = scaler[v].fit_transform(y.reshape(-1, 1)).copy()
      else:
        labels_scaled = np.hstack((labels_scaled, scaler[v].fit_transform(y.reshape(-1, 1))))
      arr = False


    inputs = create_sequences_input(X_list, y_list, tokenizer)

    print(labels_scaled)
    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(
    inputs,
    padding_side='left',
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
    )

    #input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")

    return fcllm.custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler, tokenizer, inputs, graph

def create_sequences_input(X_list, y_list, tokenizer):
  sequences = []
  for i in range(len(X_list[0])):
    seq_in = []
    seq_out = []
    for X, y in zip(X_list, y_list):
      seq_in.append(X.iloc[i].values.tolist())
      seq_out.append(float(y[i]))
    sequences.append((str(seq_in).replace("'", ""), str(seq_out)))

  text_sequences = [f"{inp} {out}" + tokenizer.eos_token for inp, out in sequences]

  return text_sequences

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_scores = self.attn(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights * x).sum(dim=1)
        return pooled

class GPT2Forecaster(nn.Module):
    def __init__(self, scaler, output_size, freeze, hidden_dims=[128, 64]):
        super().__init__()
        config = GPT2Config()
        self.gpt2 = GPT2Model(config)
        self.freeze = freeze
        if self.freeze:
            self.gpt2.requires_grad_(False)  # Freeze GPT-2 weights
        self.scaler = scaler
        self.output_size = output_size

        # Attention-based pooling over the hidden states
        self.attn_pool = AttentionPooling(config.n_embd)

        # MLP head for forecasting
        layers = []
        in_dim = config.n_embd
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels):

        gpt_outputs = self.gpt2(input_ids, attention_mask = attention_mask)
        hidden_states = gpt_outputs.last_hidden_state  # [batch_size, seq_len, embed_dim]


        pooled = self.attn_pool(hidden_states)  # [batch_size, embed_dim]

        output = self.mlp_head(pooled)  # [batch_size, output_size]



        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()

            loss = loss_fn(output, labels)


        return {"loss" : loss, "logits" : output.squeeze(0)}

def train_model(train_dataset, name_model, epochs, scaler, output_size, freeze, path_model = None):
    from transformers import TrainingArguments, Trainer
    import torch

    # Model
    model = GPT2Forecaster(scaler=scaler, output_size=output_size, freeze=freeze)

    # Enable gradient checkpointing on internal GPT2 model only
    if hasattr(model.gpt2, "gradient_checkpointing_enable"):
        model.gpt2.gradient_checkpointing_enable()
    else:
        model.gpt2.config.gradient_checkpointing = True

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.001,
        logging_steps=20,
        report_to="none",
        save_strategy='no',
        fp16=torch.cuda.is_available(),  # Mixed precision if CUDA is available
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # Optionally, you can add `eval_dataset` here
    )

    # Train the model
    trainer.train()

    # Save if path is specified
    # if path_model is not None:
    #     model.save_pretrained(path_model)

    return model


def predict(test_dataset, model, tokenizer, target, scaler, dict_variables = None):

  dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

  all_preds = pd.DataFrame()
  all_actuals = pd.DataFrame()

  for batch_data in dataloader:
    inputs = batch_data['input_ids'].to(DEVICE)
    attention_mask = batch_data['attention_mask'].to(DEVICE)
    labels = batch_data['labels'].to(DEVICE)

    outputs = model.forward(inputs, attention_mask=attention_mask, labels=None)

    predictions = outputs['logits'].detach().cpu()
    true = labels.detach().cpu()
    unscaled, trues = [], []
    for i, v in enumerate(scaler.keys()):
      unscaled.append(scaler[v].inverse_transform(predictions[:,i].reshape(-1, 1)))
      trues.append(scaler[v].inverse_transform(true[:,i].reshape(-1, 1)))

    batch_preds_df = pd.DataFrame(np.column_stack(unscaled), columns=list(scaler.keys()))
    batch_trues_df = pd.DataFrame(np.column_stack(trues), columns=list(scaler.keys()))

    all_preds = pd.concat([all_preds, batch_preds_df], ignore_index=True)
    all_actuals = pd.concat([all_actuals, batch_trues_df], ignore_index=True)

  return all_preds, all_actuals
