from transformers import BertModel, BertTokenizer, XLNetModel, XLNetTokenizer
import contextlib
import io
import sys


@contextlib.contextmanager
def suppress_stdout():
    stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = stdout


def get_model_tokenizer(model, tokenizer, pretrained_weights):
    """
    Return the model and tokenizer with the given pretrained weights.
    """
    model = model.from_pretrained(pretrained_weights)
    tokenizer = tokenizer.from_pretrained(pretrained_weights)

    return model, tokenizer


def handle_verbosity(model, tokenizer, pretrained, verbose):
    """
    Avoid duplicate code handling verbosity.
    """
    if not verbose:
        with suppress_stdout():
            model, tokenizer = get_model_tokenizer(model, tokenizer, pretrained)
    else:
        model, tokenizer = get_model_tokenizer(model, tokenizer, pretrained)

    return model, tokenizer


def get_bert_model_tokenizer(size="large", case="uncased", verbose=False):
    """
    Return the loaded BERT model and tokenizer as specified by the parameters.
    """
    assert size in ["large", "base"] and case in ["cased", "uncased"], \
        "Size must be one of \"large\" or \"base\", case must be one of " \
        "\"cased\" or \"uncased\""
    pretrained = f"bert-{size}-{case}"
    model = BertModel
    tokenizer = BertTokenizer

    return handle_verbosity(model, tokenizer, pretrained, verbose)


def get_xlnet_model_tokenizer(size="large", verbose=False):
    """
    Return the loaded XLNet model and tokenizer as specified by the parameters.
    Note that you cannot specify the casing of the model here since only the
    cased models are made available.
    """
    assert size in ["large", "base"], \
        "Size must be one of \"large\" or \"base\""

    pretrained = f"xlnet-{size}-cased"
    model = XLNetModel
    tokenizer = XLNetTokenizer

    return handle_verbosity(model, tokenizer, pretrained, verbose)
