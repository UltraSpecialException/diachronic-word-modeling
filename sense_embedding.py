from run_finetune import *
from transformers import DistilBertTokenizer
import json


def get_bert_sense_embedding(sentences, target, model, tokenizer, device):
    """
    Return the "sense embedding" by feeding sentence with the target word
    with its corresponding sense and getting the associated word emebdding.
    """
    tokenized_sentences = tokenize_inputs(
        tokenizer, sentences, add_special_tokens=True)
    tokenized_target = tokenizer.encode(target, add_special_tokens=False)[0]
    assert all(tokenized_target in tokenized_sentence
                for tokenized_sentence in tokenized_sentences)
    target_indices = [tokenized_sentence.index(tokenized_target)
                      for tokenized_sentence in tokenized_sentences]

    target_indices = torch.tensor(target_indices)

    padded_sentences = pad_inputs(tokenized_sentences, tokenizer.pad_token_id)
    padded_sentences = torch.tensor(padded_sentences)
    masks = get_padding_mask(padded_sentences)
    padded_sentences = padded_sentences.to(device)
    masks = masks.to(device)

    bert = model.distilbert
    hidden_states = bert(padded_sentences, attention_mask=masks)[0]

    shape = hidden_states.size()
    hidden_states = hidden_states.view(-1, shape[-1])
    target_indices += torch.arange(0, shape[0] * shape[1], shape[1])
    sense_embedding = hidden_states[target_indices].mean(dim=0)
    sense_embedding = sense_embedding.to(torch.device("cpu"))

    return sense_embedding


if __name__ == "__main__":
    pretrained_model = "distilbert-base-uncased"
    num_layers = 2
    activation = "gelu"
    device = torch.device("cpu")
    print(f"Using pretrained model: {pretrained_model}")
    bert_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
    model = DistilBertWordSenseDisambiguation.from_pretrained(
        pretrained_model,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.reset_classifier(num_layers, activation)
    bert_model = model

    target_words = [
        "face", "part", "head", "record", "word", "edge", "land",
        "circle", "relationship", "rag", "fiction", "ball", "plane",
        "risk", "gas", "ounce", "bag", "prop", "bit", "tree", "twist",
        "attack", "savage", "tip", "pin", "player", "contemplation",
        "lane", "stroke", "thump", "stab", "chairman"
    ]

    sentences_and_word_senses = json.load(open(
        "words_with_sentences_separated_by_senses.json"
    ))

    sense_embeddings = {word: {} for word in target_words}
    for word in sentences_and_word_senses:
        print("Processing word", word)
        for sense in sentences_and_word_senses[word]:
            embedding = torch.zeros(768)
            for sentence in sentences_and_word_senses[word][sense]:
                sentences = [sentence]
                embedding += get_bert_sense_embedding(
                    sentences, word, bert_model, bert_tokenizer, device)

            embedding = embedding.mean(dim=0)
            sense_embeddings[word][sense] = embedding
