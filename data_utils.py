import os
import requests
import json
from collections import defaultdict
import re

corpus1_path = "project/data/semeval2020_task1_eng/corpus1/lemma/ccoha1.txt"
corpus2_path = "project/data/semeval2020_task1_eng/corpus2/lemma/ccoha2.txt"
target_path = "project/data/semeval2020_task1_eng/targets.txt"

paths = {
    1: os.path.join(os.path.abspath("."), corpus1_path),
    2: os.path.join(os.path.abspath("."), corpus2_path)
}


def save_json(path, data):
    """
    Save a json at the given path using the data.
    """
    with open(path, "w+") as data_file:
        json.dump(data, data_file)


def open_corpus(corpus_num, as_list=True):
    """
    Read the corpus and return it as an IO stream.
    """
    data = open(paths[corpus_num])
    return data if not as_list else data.readlines()


def view_first_lines(corpus, num_lines):
    """
    Print out the first <num_lines> of the corpus.

    If the corpus is not passed in as a list of sentences, the function will
    attempt to read the corpus using .readlines().
    """
    if not isinstance(corpus, list):
        corpus = corpus.readlines()

    for line in corpus[:num_lines]:
        print(line)


def get_target_sentences(corpus, save_path=None):
    """
    From the given corpus, extract sentences that contain the target words.
    """
    data = open_corpus(corpus)
    word_to_sentence = defaultdict(list)
    for line in data:
        if "_nn" in line or "_vb" in line:
            words = re.findall(r"([^\s]*_nn|[^\s]*_vb)", line)
            line = re.sub(r"(_nn|_vb)", "", line)

            for word in words:
                if word[:-3].strip() != "xx":
                    word_to_sentence[word[:-3].strip()].append(line.strip())

    if save_path is not None:
        print("Saving target sentences.")
        save_json(save_path, word_to_sentence)

    return word_to_sentence


def retrieve_targets():
    """
    Return a list of tuple of (target word, POS) that is expected.
    """
    pos = {
        "nn": "noun",
        "vb": "verb"
    }

    file = open(target_path).readlines()
    targets = []
    for line in file:
        word, tag = line.strip().split("_")
        targets.append((word, pos[tag]))

    return targets


def retrieve_definitions(target_words, save_path=None):
    """
    Use the Oxford Dictionaries API to retrieve associated definitions.
    """
    try:
        app_id = os.environ["APP_ID"]
        app_key = os.environ["APP_KEY"]
    except KeyError:
        raise KeyError("Environment variables APP_ID and APP_KEY are necessary "
                       "to make the api calls to OED. Please set them.")

    data = []
    for word, _ in target_words:
        url = f"https://od-api.oxforddictionaries.com/api/v2/entries/en-us/" \
            f"{word}?fields=definitions&strictMatch=false"
        result = requests.get(url,
                              headers={"app_id": app_id, "app_key": app_key})

        if str(result.status_code) != "200":
            raise RuntimeError(f"Error code {result.status_code} when getting "
                               f"definition for word {word}, "
                               f"error message: {result.text}")

        print(f"Retrieved definition(s) for word {word} successfully!")
        data.append(result.json())

    if save_path is not None:
        print("Saving data to JSON file.")
        with open(save_path, "w+") as data_file:
            json.dump(data, data_file)

    return data


def parse_senses(definition_responses, target_words, save_path=None):
    """
    Return a list of senses given the
    """
    target_words = {word: pos for word, pos in target_words}

    assert isinstance(definition_responses, str) \
           or isinstance(definition_responses, list), \
        "Argument defintion_reponses need to either be a path to json file " \
        "or a list of definitions"

    if isinstance(definition_responses, str):
        definition_responses = json.load(open(definition_responses))

    data = {}

    for item in definition_responses:
        word = item["id"]
        target_pos = target_words[word]

        item_data = {"senses": [], "subsenses": []}

        for result in item["results"]:
            senses = get_senses_from_lexical_entries(result, target_pos)
            item_data["senses"].extend(senses["senses"])
            item_data["subsenses"].extend(senses["subsenses"])

        data[word] = item_data

    if save_path is not None:
        print("Saving cleaned data...")
        with open(save_path, "w+") as data_file:
            json.dump(data, data_file)

    return data


def get_senses_from_lexical_entries(result, target_pos):
    """
    Helper function to get senses from the results requested for a word.
    """
    lexical_entries = result["lexicalEntries"]
    all_senses_dict = {"senses": [], "subsenses": []}
    for lex_ent in lexical_entries:
        if lex_ent["lexicalCategory"]["id"] == target_pos:
            for entry in lex_ent["entries"]:
                senses = get_senses(entry)
                all_senses_dict["senses"].extend(senses["senses"])
                all_senses_dict["subsenses"].extend(senses["subsenses"])

    return all_senses_dict


def get_senses(lexical_entry):
    """
    Format and return a list of senses
    """
    senses_dict = defaultdict(list)
    senses = lexical_entry["senses"] if "senses" in lexical_entry else []

    for sense in senses:
        try:
            senses_dict["senses"].append(sense["definitions"])
        except KeyError:
            senses_dict["senses"].append([])

        subsenses = sense["subsenses"] if "subsenses" in sense else []

        for subsense in subsenses:
            try:
                senses_dict["subsenses"].append(subsense["definitions"])
            except KeyError:
                senses_dict["subsenses"].append([])

    return senses_dict


class DataCollection:
    """
    Organizes the data into batches for easier training.
    """

    def __init__(self, senses_inventory, data, batch_size):
        """
        Initializes the batch object.
        """
        self.senses_inventory = senses_inventory
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        pass


class DataIterator:
    """
    Iterator for class DataIterator.
    """

    def __init__(self, batches):
        self.batches = batches
        self.next_batch = 1

    def __next__(self):
        if self.next_batch > len(self.batches):
            raise StopIteration

        batch = self.batches[self.next_batch - 1]
        self.next_batch += 1

        return batch
