import re
from string import punctuation
import numpy as np


def remove_extra_spaces(string):
    return string


# Attempts to fix instances where the beginning of the instance is strange
def remove_leading_spaces(string):
    regex = r"^ +"

    test_str = string

    subst = ""

    result = re.sub(regex, subst, test_str, 0, re.MULTILINE)

    if result:
        return result

    return string


def remove_leading_colons(string):
    regex = r"^:+"

    test_str = string

    subst = ""

    result = re.sub(regex, subst, test_str, 0, re.MULTILINE)

    if result:
        return result

    return string


def remove_leading_apostrophes(string):
    regex = r"^'+"

    test_str = string

    subst = ""

    result = re.sub(regex, subst, test_str, 0, re.MULTILINE)

    if result:
        return result

    return string


def remove_leading_carats(string):
    regex = r"^`+"

    test_str = string

    subst = ""

    result = re.sub(regex, subst, test_str, 0, re.MULTILINE)

    if result:
        return result

    return string


def remove_hyperlinks(string):
    regex = r"http?://\S+|www\.\S+"

    test_str = string

    subst = ""

    # You can manually specify the number of replacements by changing the 4th argument
    result = re.sub(regex, subst, test_str, 0, re.MULTILINE)

    if result:
        return result

    else:
        return string


def remove_common_contractions(string):
    with open('common_contractions.txt', 'r') as fp:
        contractions = fp.readlines()
        contractions[:] = [x.lower().strip() for x in contractions]
        contractions[:] = [x.replace('’', "'") for x in contractions]
        fp.close()

    temp_string = [string]

    for item in contractions:
        curr_item = item.split(";")
        temp_string[0] = temp_string[0].replace(curr_item[0], curr_item[1])

    return temp_string[0]


def convert_to_lowercase(string):
    return string.lower()


# Below was borrowed from top rated answer by 'Brian' https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
def remove_punctuation(string):
    return string.translate(str.maketrans('', '', punctuation))


def preprocess_user_comment(string):
    processed_string = remove_extra_spaces(string)
    processed_string = remove_leading_spaces(processed_string)
    processed_string = remove_leading_colons(processed_string)
    processed_string = remove_leading_apostrophes(processed_string)
    processed_string = remove_leading_carats(processed_string)
    processed_string = remove_hyperlinks(processed_string)
    processed_string = remove_common_contractions(processed_string)
    processed_string = processed_string.lower()
    processed_string = remove_punctuation(processed_string)

    return processed_string


# BERT only allows up to 512 tokens per sentence (including [CLS] [SEP] tokens)
def trim_sent_length(string):
    if len(string) > 500:
        truncated_sent = string[0:500]
        return truncated_sent

    else:
        return string


def tokenize_user_comment(string, tokenizer):
    return tokenizer(string, return_tensors='pt')


# Generate a single 1x768 pooled embedding for a user comment - this should be able to be converted to numpy array and used in model
def generate_comment_embedding(tokenized_string, embedding_model):
    output = embedding_model(**tokenized_string)
    pooled_embedding = output[1][0]
    return pooled_embedding


def extract_embedding_values(embedding):
    embedding = embedding.tolist()
    embedding = np.array(embedding)
    return embedding
