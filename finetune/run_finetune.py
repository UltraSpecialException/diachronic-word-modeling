import os
import sys

sys.path.append(os.path.abspath("../"))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, \
    RandomSampler, SequentialSampler
import argparse
from modules import *
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from transformers import BertForSequenceClassification, \
    get_linear_schedule_with_warmup, AdamW,\
    DistilBertForSequenceClassification, XLNetForSequenceClassification


def tokenize_inputs(tokenizer, sentences, add_special_tokens=True):
    """
    Use the tokenizer given to tokenize the sentences into their IDs.
    """
    print("Tokenizing sentences...")
    tokenized_sentences = []
    progress = tqdm(total=len(sentences))
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(
            sentence,
            add_special_tokens=add_special_tokens
        )
        tokenized_sentences.append(tokenized_sentence)
        progress.update(1)

    return tokenized_sentences


def pad_inputs(tokenized_sentences, padding_token=0):
    """
    Return the padded sentences where each sentence is padded with 0's so that
    all sentences have the length of the longest sentence.
    """
    max_len = max([len(sentence) for sentence in tokenized_sentences])

    return pad_sequences(tokenized_sentences, maxlen=max_len, dtype="long",
                         value=padding_token, truncating="post", padding="post")


def get_padding_mask(padded_sentences):
    """
    Return a list of masks, one for each tokenized sentence, where at each
    position of the sentence, if the token has a non-zero value, then the token
    has meaning and thus is not a padding token and will have value 1 in the
    corresponding position in the mask, otherwise, the token is a padding token
    and will have value 0 in the corresponding position in the mask.
    """
    return padded_sentences > 0


def get_train_val_loader(inputs, masks, labels, batch_size, train_split=0.8):
    """
    Return the train and validation data loader.
    """
    assert 0 < train_split < 1, \
        "train_split needs to be a fraction between 0 and 1 exclusive"
    num_train = int(np.ceil(inputs.size(0) * train_split))
    num_val = int(inputs.size(0) - num_train)

    assert num_train and num_val, \
        f"the train_split given ({train_split}) resultted in either the " \
        f"number of training or validation examples being 0, which is " \
        f"invalid"

    train_indices = torch.randint(low=0, high=inputs.size(0), size=(num_train,))
    used = set([int(num) for num in train_indices])
    val_indices = torch.tensor(
        [i for i in range(inputs.size(0)) if i not in used]
    )

    train_inputs, train_masks, train_labels = \
        inputs[train_indices], masks[train_indices], labels[train_indices]

    val_inputs, val_masks, val_labels = inputs[val_indices], \
                                        masks[val_indices], labels[val_indices]

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size, sampler=train_sampler)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, batch_size, sampler=val_sampler)

    return train_dataloader, val_dataloader


name_to_activation = {
        "relu": nn.ReLU,
        "gelu": nn.GELU
    }


class BertWordSenseDisambiguation(BertForSequenceClassification):
    """
    Apply BERT for Word Sense Disambiguation task.
    """

    def __init__(self, config):
        """
        Initialize an instance of WSD BERT.
        """
        super(BertWordSenseDisambiguation, self).__init__(config)
        self.config_ = config

    def reset_classifier(self, num_layers, activation):
        """
        Reset the classifier to default initialization.
        """
        config = self.config_
        blocks = []
        for _ in range(num_layers):
            layer = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            blocks.append(layer)
            blocks.append(name_to_activation[activation]())

        self.classifier = nn.Sequential(
            *blocks,
            nn.Linear(config.hidden_size, self.config.num_labels)
        )


class DistilBertWordSenseDisambiguation(DistilBertForSequenceClassification):
    """
    Apply BERT for Word Sense Disambiguation task.
    """

    def __init__(self, config):
        """
        Initialize an instance of WSD BERT.
        """
        super(DistilBertWordSenseDisambiguation, self).__init__(config)
        self.config_ = config

    def reset_classifier(self, num_layers, activation):
        """
        Reset the classifier to default initialization.
        """
        config = self.config_
        blocks = []
        for _ in range(num_layers):
            layer = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            blocks.append(layer)
            blocks.append(name_to_activation[activation]())

        self.classifier = nn.Sequential(
            *blocks,
            nn.Linear(config.hidden_size, self.config.num_labels)
        )


class XLNetWordSenseDisambiguation(XLNetForSequenceClassification):
    """
    Apply XLNet for Word Sense Disambiguation task.
    """

    def __init__(self, config):
        """
        Initialize an instance of WSD BERT.
        """
        super(XLNetWordSenseDisambiguation, self).__init__(config)
        self.config_ = config

    def reset_classifier(self, num_layers, activation):
        """
        Reset the classifier to default initialization.
        """
        config = self.config_
        blocks = []
        for _ in range(num_layers):
            layer = nn.Linear(config.d_model, config.d_model)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            blocks.append(layer)
            blocks.append(name_to_activation[activation]())

        self.logits_proj = nn.Sequential(
            *blocks,
            nn.Linear(config.d_model, self.config.num_labels)
        )


def check_path(path):
    full_path = path
    if full_path[-1] == "/":
        full_path = full_path[:-1]

    parent_end = full_path.rfind("/")
    if parent_end == -1:
        parent_end = 0
    parent = os.path.join(
        os.path.abspath("."),
        full_path[:parent_end]
    )
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"parent path {parent} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSD Finetuning scripts")

    parser.add_argument("-data_path",
                        help="Path to the data",
                        required=True)

    parser.add_argument("--model",
                        help="The model to finetune, choose from bert & xlnet",
                        choices=["bert", "xlnet"],
                        default="bert",
                        type=str)

    parser.add_argument("--bert_case",
                        help="Whether to use the case or uncased version of "
                             "BERT",
                        choices=["cased", "uncased"],
                        default="uncased",
                        type=str)

    parser.add_argument("--bert_size",
                        help="Whether to use the base or large size of BERT",
                        choices=["base", "large"],
                        default="large",
                        type=str)

    parser.add_argument("--xlnet_size",
                        help="Whether to use the base or large size of XLNet",
                        choices=["base", "large"],
                        default="large",
                        type=str)

    parser.add_argument("--num_layers",
                        help="The number of layers to use for the classifier",
                        default=2,
                        type=int)

    parser.add_argument("--activation",
                        help="The activation function to use in the classifier",
                        choices=["gelu", "relu"],
                        default="gelu",
                        type=str)

    parser.add_argument("--batch_size",
                        help="The batch size for fine-tuning",
                        default=64,
                        type=int)

    parser.add_argument("--epochs",
                        help="Number of epochs to train for",
                        default=1,
                        type=int)

    parser.add_argument("--seed",
                        help="Random seed for reproducibility",
                        default=42,
                        type=int)

    parser.add_argument("--no_cuda",
                        help="Do not use GPU",
                        action="store_true")

    parser.add_argument("--parallel",
                        help="Parallelize over multiply GPUs",
                        action="store_true")

    parser.add_argument("--padded_data_input",
                        help="Path to input file of saved padded data")

    parser.add_argument("--weights_output",
                        help="Path to output file to save weights")

    parser.add_argument("--padded_data_output",
                        help="Path to output file to save padded data")

    args = parser.parse_args()

    if args.weights_output is not None:
        check_path(args.weights_output)

    if args.padded_data_input is not None:
        check_path(args.padded_data_input)
    elif args.padded_data_output is not None:
        check_path(args.padded_data_output)

    if args.model == "bert":
        pretrained_model = f"bert-{args.bert_size}-{args.bert_case}"
        print(f"Using pretrained model: {pretrained_model}")
        model = BertWordSenseDisambiguation.from_pretrained(
            pretrained_model,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        # handle XLNet
        pretrained_model = f"xlnet-{args.xlnet_size}-cased"
        model = XLNetWordSenseDisambiguation.from_pretrained(
            pretrained_model,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )

    model.reset_classifier(args.num_layers, args.activation)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
        model = model.to(device)

        if torch.cuda.device_count() > 1 and args.parallel:
            model = nn.DataParallel(model)

    plain_data = open(args.data_path).readlines()[:]

    sentences = []
    labels = []
    for line in plain_data:
        sentence, label = line.strip().split("@")
        sentences.append(sentence)
        labels.append(int(label))

    if args.padded_data_input is not None:
        print("Loading padded data...")
        padded_data = torch.load(args.padded_data_input, map_location="cpu")
        print("Data loaded...")
    else:
        if args.model == "bert":
            _, tokenizer = get_bert_model_tokenizer(args.bert_size,
                                                    args.bert_case)
        else:
            _, tokenizer = get_xlnet_model_tokenizer(args.xlnet_size)

        tokenized_data = tokenize_inputs(tokenizer, sentences)
        padded_data = pad_inputs(tokenized_data, tokenizer.pad_token_id)
        padded_data = torch.tensor(padded_data)
        if args.padded_data_output is not None:
            torch.save(padded_data, args.padded_data_output)

    # might want to transfer to GPU later so it doesn't crowd the GPU up
    padded_data = padded_data.to(device)
    labels = torch.tensor(labels).to(device)
    masks = get_padding_mask(padded_data).to(device)

    print("Getting dataloaders...")
    train_dataloader, val_dataloader = get_train_val_loader(
        padded_data,
        masks,
        labels,
        args.batch_size
    )

    print("Setting up...")
    total_iterations = len(train_dataloader) * args.epochs

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_iterations)

    for epoch in range(args.epochs):
        print(f"Starting epoch #{epoch + 1} of {args.epochs}")
        train_progress = tqdm(total=len(train_dataloader))
        model.train()
        for inputs, masks, labels in train_dataloader:
            model.zero_grad()
            outputs = model(
                inputs,
                token_type_ids=None,
                attention_mask=masks,
                labels=labels
            )

            loss = outputs[0]
            train_progress.set_description('[Loss: {:.4f}]'.format(loss.item()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_progress.update(1)
            del loss
            torch.cuda.empty_cache()

        train_progress.close()

        print(f"Validating...")
        val_progress = tqdm(total=len(val_dataloader))
        model.eval()
        correct = 0
        total = 0
        for inputs, masks, labels in val_dataloader:
            with torch.no_grad():
                outputs = model(
                    inputs,
                    token_type_ids=None,
                    attention_masks=masks,
                    labels=labels
                )

            loss, logits = outputs[:2]
            val_progress.set_description('[Loss: {:.4f}]'.format(loss.item()))

            preds = logits.argmax(dim=1).view(-1)
            # noinspection PyUnresolvedReferences
            correct = (preds == labels).sum()
            total += preds.size(0)
            val_progress.update(1)

        print(f"Validation accuracy: {correct/total}")

    torch.save(model.state_dict(), args.weights_output)
