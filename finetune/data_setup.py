import re
from project.data_utils import save_json
import requests
from bs4 import BeautifulSoup as bs
import json
from collections import defaultdict


def extract_semcor(data_path, words_output_path, sentences_output_path):
    """
    Using the Semcor XML data format, extract the target words.
    """
    with open(data_path) as data_file:
        data = data_file.readlines()[:]

        words = []
        sentence = []
        sentences_to_word = {}
        current_words = []

        for line in data:
            target_word = re.findall("(?<=lemma=\").*(?=\" )", line.strip())
            sentence.extend(target_word)
            if line[1:].startswith("instance"):
                words.extend(target_word)
                id = re.findall("(?<=id=\").*(?=\" lemma)", line.strip())
                word_and_id = (target_word[0], id[0])
                current_words.append(word_and_id)

            elif line[1:].startswith("/sentence"):
                sentences_to_word[" ".join(sentence)] = current_words
                sentence = []
                current_words = []

        words = list(set(words))
        data_file.close()

    with open(words_output_path, "w+") as words_output_file:
        for word in words:
            words_output_file.write(f"{word}\n")

        words_output_file.close()

    save_json(sentences_output_path, sentences_to_word)

    return words


def get_word_net_url(word):
    """
    Return the url for the request call to WordNet by Princeton.
    """
    return f"http://wordnetweb.princeton.edu/perl/webwn?s={word}&sub=Search+" \
        f"WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=1&o3=&o4=&h=0"


key_regex = [r"[a-z-_']*%[0-9]:[0-9]{2}:[0-9]{2}::",
             r"[a-z-_']*%[0-9]:[0-9]{2}:[0-9]{2}:[a-z-_]*:[0-9]{2}"]


meaning_regex = [r"(?<=:{2}\) \().*(?=\))", r"(?<=[0-9]{2}\) \().*(?=\))"]


def extract_word_senses(word_list, save_path=None):
    """
    From the word list given, return a dictionary of each word and their
    corresponding sense keys and sense meanings.
    """
    data = {}
    temp_data = {}
    for i, word in enumerate(word_list):
        word = word.replace("_", "+")
        word = word.replace("&apos;", "'")
        print(f"Processing word #{i + 1}: {word}")
        data[word] = {}
        temp_data[word] = {}
        url = get_word_net_url(word)
        result = requests.get(url)
        soup = bs(result.content, "html.parser")
        all_senses = soup.find_all("li")

        for sense in all_senses:
            sense = sense.get_text()
            keys = []

            for regex in key_regex:
                senses_keys = re.findall(regex, sense)
                keys.extend(senses_keys)

            meanings = re.findall(meaning_regex[0], sense)
            if meanings:
                meaning = meanings[0]
            else:
                meaning = re.findall(meaning_regex[1], sense)[0]

            for key in keys:
                data[word][key] = meaning
                temp_data[word][key] = meaning

        if not ((i + 1) % 1000) or i + 1 == len(word_list):
            save_json(f"../data/semcor_key_to_sense_pos.json", temp_data)
            temp_data = {}

    if save_path is not None:
        save_json(save_path, data)


def format_sentence(sentence, target_word, sense):
    """
    Return the sentence as described in the format_data function.
    """
    return f"The word {target_word} in the sentence \" {sentence} \" has the " \
        f"sense \" {sense} \" ."


def format_data(sentence_to_targets, senses_inventory, senses_id_to_meaning,
                labeled_data, save_path=None):
    """
    Format the data for classification purpose.

    The classification task will be binary. We will format the data in a form
    that explicitly states the sense of the target word of a sentence.

    Essentially, we will format our sentences to the form:
    "The word <target_word> in the sentence '<sentence>' has the sense <sense>"
    """
    lines = defaultdict(list)
    error_words = set()
    for sentence in sentence_to_targets:
        sentence_ = sentence.replace("&apos;", "'")
        for lemma, lemma_id in sentence_to_targets[sentence]:
            lemma = lemma.replace("&apos;", "'")
            for sense in senses_inventory[lemma]:
                formatted_sentence = format_sentence(sentence_, lemma, sense)
                label = "0"
                for key in labeled_data[lemma_id]:
                    try:
                        if sense == senses_id_to_meaning[lemma][key]:
                            label = "1"
                            break
                    except KeyError:
                        error_words.add(lemma)
                        continue

                lines[lemma].append("@".join([formatted_sentence, label]))

    with open(save_path, "w+") as data_file:
        for word in lines:
            if word not in error_words:
                for line in lines[word]:
                    data_file.write(line + "\n")

        data_file.close()

    return lines


if __name__ == "__main__":
    sentence_to_targets = json.load(open("../data/semcor_sentence_to_words.json"))
    senses_inventory = json.load(open("../data/semcor_word_to_all_senses.json"))
    senses_id_to_meaning = json.load(open("../data/semcor_key_to_sense.json"))
    labeled_data = json.load(open('../data/semcor_lemma_id_to_labels.json'))
    format_data(sentence_to_targets, senses_inventory, senses_id_to_meaning, labeled_data, "../data/semcor_training_data.txt")

