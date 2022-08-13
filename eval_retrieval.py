# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import nltk
import json
import os
import logging
import sys
import time
import string
import random

ENT_START = "[START_ENT]"
ENT_END = "[END_ENT]"


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_punc(lower(s))


def validate_datapoint(datapoint, logger):

    # input is a string
    if not isinstance(datapoint["input"], str):
        if logger:
            logger.warning(
                "[{}] input is not a string {}".format(
                    datapoint["id"], datapoint["input"]
                )
            )
        return False

    # output is not empty
    if "output" in datapoint:
        if len(datapoint["output"]) == 0:
            if logger:
                logger.warning("[{}] empty output".format(datapoint["id"]))
            return False

        for output in datapoint["output"]:
            # answer is a string
            if "answer" in output:
                if not isinstance(output["answer"], str):
                    if logger:
                        logger.warning(
                            "[{}] answer is not a string {}".format(
                                datapoint["id"], output["answer"]
                            )
                        )
                    return False

            # provenance is not empty
            # if len(output["provenance"]) == 0:
            #    if logger:
            #        logger.warning("[{}] empty provenance".format(datapoint["id"]))
            #    return False

            if "provenance" in output:
                for provenance in output["provenance"]:
                    # wikipedia_id is provided
                    if provenance["wikipedia_id"] is not None and not isinstance(
                        provenance["wikipedia_id"], str
                    ):
                        if logger:
                            logger.warning(
                                "[{}] wikipedia_id is not a string {}".format(
                                    datapoint["id"], provenance["wikipedia_id"]
                                )
                            )
                        return False

                    # title is provided
                    if not isinstance(provenance["title"], str):
                        if logger:
                            logger.warning(
                                "[{}] title is not a string {}".format(
                                    datapoint["id"], provenance["title"]
                                )
                            )
                        return False

    return True


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def store_data(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")


def get_bleu(candidate_tokens, gold_tokens):

    candidate_tokens = [x for x in candidate_tokens if len(x.strip()) > 0]
    gold_tokens = [x for x in gold_tokens if len(x.strip()) > 0]

    # The default BLEU calculates a score for up to
    # 4-grams using uniform weights (this is called BLEU-4)
    weights = (0.25, 0.25, 0.25, 0.25)

    if len(gold_tokens) < 4:
        # lower order ngrams
        weights = [1.0 / len(gold_tokens) for _ in range(len(gold_tokens))]

    BLEUscore = nltk.translate.bleu_score.sentence_bleu(
        [candidate_tokens], gold_tokens, weights=weights
    )
    return BLEUscore


# split a list in num parts evenly
def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num  # 0 <= diff < num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def init_logging(base_logdir, modelname, logger=None):

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_directory = "{}/{}/".format(base_logdir, modelname)

    if logger == None:
        logger = logging.getLogger("KILT")

        logger.setLevel(logging.DEBUG)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    else:
        # remove previous file handler
        logger.handlers.pop()

    os.makedirs(log_directory, exist_ok=True)

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.propagate = False
    logger.info("logging in {}".format(log_directory))
    return logger


def create_logdir_with_timestamp(base_logdir):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, timestr, random.randint(0, 1000))
    os.makedirs(log_directory)
    return log_directory


def match_answer(
    answer,
    page,
    nlp=None,
    MAX_PARAGRAPH_CANDIDATE=3,
    debug=False,
    index_mapping=None,
    normalize_text=True,
    fast=False,
    approximate_search=False,
):
    # if nlp == None:
    #    nlp = spacy.load("en_core_web_sm")

    original_answer = answer
    if normalize_text:
        answer = normalize_answer(answer)

    try:
        if nlp == None or approximate_search:
            answer_tokens = [token for token in answer.split()]
        else:
            answer_tokens = [token.text for token in nlp(answer)]
    except Exception as e:
        print("Exception {}".format(e))
        return -1, -1, -1, -1

    if normalize_text:
        # Remove “characters with encodings larger than 3 bytes” using Python 3
        answer_tokens = [
            "".join(char for char in x if len(char.encode("utf-8")) < 3).lower()
            for x in answer_tokens
        ]

    found = False
    max_bleu = None
    start_token = None
    end_token = None
    paragraph_id = None

    # instead of scanning all lines, get the k with the higest intersection
    candidate_dict = {}
    tokenized_paragraphs = []
    tokenized_paragraphs_offset = []

    for idx, paragraph in enumerate(page["text"]):

        index = paragraph.find(answer)
        if index >= 0:
            assert paragraph[index : index + len(answer)] == answer
            return idx, index, index + len(original_answer), 1.0

        index = paragraph.find(original_answer)
        if index >= 0:
            assert paragraph[index : index + len(original_answer)] == original_answer
            return idx, index, index + len(original_answer), 1.0

        paragraph_tokens = []
        paragraph_offsets = []

        if nlp == None or approximate_search:
            seen = ""
            for token in paragraph.split():
                paragraph_tokens.append(token)
                paragraph_offsets.append(0)  # offset are unreliable without nlp
                seen += str(token) + " "
        else:
            for token in nlp(paragraph):
                paragraph_tokens.append(token.text)
                # idx	int	The character offset of the token within the parent document.
                paragraph_offsets.append(token.idx)

        if normalize_text:
            # Remove “characters with encodings larger than 3 bytes” using Python 3
            paragraph_tokens = [
                normalize_answer(
                    "".join(char for char in x if len(char.encode("utf-8")) < 3)
                )
                for x in paragraph_tokens
            ]

        tokenized_paragraphs.append(paragraph_tokens)
        tokenized_paragraphs_offset.append(paragraph_offsets)

        # token intersection
        intersection = len(set(paragraph_tokens).intersection(set(answer_tokens)))

        if intersection == len(answer_tokens):
            # I found all the tokens, let me see if there is a perfect match
            ax = " ".join([x.strip() for x in answer_tokens if len(x.strip()) > 0])
            for w_start in range(len(paragraph_tokens)):
                token = paragraph_tokens[w_start]
                if token == answer_tokens[0]:
                    bx = " ".join(
                        [
                            x.strip()
                            for x in paragraph_tokens[w_start:]
                            if len(x.strip()) > 0
                        ]
                    )
                    if bx.startswith(ax):
                        for w_end in range(w_start, len(paragraph_tokens)):
                            token = paragraph_tokens[w_end]
                            if token == answer_tokens[-1]:
                                cx = " ".join(
                                    [
                                        x.strip()
                                        for x in paragraph_tokens[w_start : w_end + 1]
                                        if len(x.strip()) > 0
                                    ]
                                )
                                if ax == cx:
                                    start_character = paragraph_offsets[w_start]
                                    end_character = paragraph_offsets[w_end] + len(
                                        paragraph_tokens[w_end]
                                    )
                                    return idx, start_character, end_character, 1.0

        if intersection not in candidate_dict:
            candidate_dict[intersection] = []
        candidate_dict[intersection].append(idx)

    candidate_idx = []
    for key in sorted(candidate_dict.keys(), reverse=True):
        # if key > 0:  # if the intersection is not empty
        for idx in candidate_dict[key]:
            candidate_idx.append(idx)
        if len(candidate_idx) >= MAX_PARAGRAPH_CANDIDATE:
            break

    assert len(candidate_idx) > 0

    # hack to map to new knowledge source
    if index_mapping:
        new_candidate_idx = []
        for idx in candidate_idx:
            if idx not in index_mapping:
                new_candidate_idx.append(idx)
        candidate_idx = new_candidate_idx
        if len(candidate_idx) == 0:
            return -1, -1, -1, -1

    if fast:
        return candidate_idx[0], -1, -1, -1

    if nlp != None and approximate_search:
        # now get the proper tokenized version for the candidate idx and answer
        answer_tokens = [token.text for token in nlp(answer)]
        for idx in candidate_idx:
            paragraph_tokens = []
            paragraph_offsets = []
            for token in nlp(page["text"][idx]):
                paragraph_tokens.append(token.text)
                # idx	int	The character offset of the token within the parent document.
                paragraph_offsets.append(token.idx)
            tokenized_paragraphs[idx] = paragraph_tokens
            tokenized_paragraphs_offset[idx] = paragraph_offsets

    # then scan only the k candidates
    for idx in candidate_idx:

        paragraph_tokens = tokenized_paragraphs[idx]

        # perfect match
        for i in range(len(paragraph_tokens) - len(answer_tokens) + 1):
            if paragraph_tokens[i : i + len(answer_tokens)] == answer_tokens:
                found = True
                max_bleu = 1.0
                paragraph_id = idx
                start_token = i
                end_token = i + len(answer_tokens)
                break

        # fuzzy match
        if not found:

            # TODO: add span tollerance to speed up! Not sure about this
            # SPAN_TOLLERANCE = int(len(answer_tokens) / 2)

            for init in range(len(paragraph_tokens)):
                for end in range(init, len(paragraph_tokens)):
                    candidate = paragraph_tokens[init : end + 1]
                    BLEU = get_bleu(candidate, answer_tokens)

                    # if there is the same BLEU, the shortest answer should win
                    if (
                        not max_bleu
                        or BLEU > max_bleu
                        or (
                            BLEU == max_bleu
                            and end_token
                            and start_token
                            and (end + 1 - init) < (end_token - start_token)
                        )
                    ):
                        max_bleu = BLEU
                        paragraph_id = idx
                        start_token = init
                        end_token = end

                    if max_bleu == 1:
                        break
                if max_bleu == 1:
                    break
            if max_bleu == 1:
                break

    if debug:
        print("wikipedia_tile:", page["wikipedia_title"])
        print("bleu: {0:.2f}".format(max_bleu))
        print("paragraph_id:", paragraph_id)
        print("start_token_id:", start_token)
        print("end_token_id:", end_token)
        print("start_token:", tokenized_paragraphs[paragraph_id][start_token])
        print("end_token:", tokenized_paragraphs[paragraph_id][end_token])
        print(
            "TOKENIZED MATCH", tokenized_paragraphs[paragraph_id][start_token:end_token]
        )
        print("len(tokenized_paragraphs):", len(tokenized_paragraphs))
        print("len(tokenized_paragraphs_offset):", len(tokenized_paragraphs_offset))
        print("paragraph_tokens:", tokenized_paragraphs[paragraph_id])
        print("paragraph_offsets:", tokenized_paragraphs_offset[paragraph_id])
        print(
            "start_character:", tokenized_paragraphs_offset[paragraph_id][start_token]
        )
        print("end_character:", tokenized_paragraphs_offset[paragraph_id][end_token])

    paragraph_tokens = tokenized_paragraphs[paragraph_id]
    paragraph_offsets = tokenized_paragraphs_offset[paragraph_id]

    if nlp == None:
        # offset are unreliable without nlp
        start_character = -1
        end_character = -1
    else:
        start_character = paragraph_offsets[start_token]
        end_character = paragraph_offsets[end_token] + len(paragraph_tokens[end_token])

    return paragraph_id, start_character, end_character, max_bleu


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import pprint
import re
import string
from rouge import Rouge

from collections import Counter

# utility to get gold answers
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths


# utility to get gold titles
def get_gold_titles(gold):
    titles = set()
    for item in gold["output"]:
        if "provenance" in item:
            for provenance in item["provenance"]:
                if (
                    "title" in provenance
                    and provenance["title"]
                    and len(provenance["title"].strip()) > 0
                ):
                    titles.add(provenance["title"].strip())
    return titles


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _calculate_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    # kilt metrics
    kilt_accuracy = 0
    kilt_em = 0
    kilt_f1 = 0
    kilt_rougel = 0

    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs"

        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        conditions = (len(guess_item["output"]) == 1) and (
            "answer" in guess_item["output"][0]
        )
        assert (
            conditions
        ), f"you should provide exactly one valid answer for {guess_item['id']}"
        guess_answer = str(guess_item["output"][0]["answer"]).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )
        normalized_f1 += local_f1

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
        rougel += local_rougel

        # KILT-metrics
        Rprec = retrieval_metrics.rprecision(
            guess_item, gold_item, rank_keys=["wikipedia_id"]
        )
        if Rprec == 1:
            # 1. KILT-AC
            kilt_accuracy += local_accuracy

            # 2. KILT-EM
            kilt_em += local_em

            # 3. KILT-F1
            kilt_f1 += local_f1

            # 4. KILT-RL
            kilt_rougel += local_rougel

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        kilt_accuracy /= total_count
        kilt_em /= total_count
        kilt_f1 /= total_count
        kilt_rougel /= total_count

    return {
        "kilt": {
            "KILT-accuracy": kilt_accuracy,
            "KILT-em": kilt_em,
            "KILT-f1": kilt_f1,
            "KILT-rougel": kilt_rougel,
        },
        "downstream": {
            "accuracy": accuracy,
            "em": normalized_em,
            "f1": normalized_f1,
            "rougel": rougel,
        },
    }


def validate_input(gold_records, guess_records):

    if len(gold_records) != len(guess_records):
        print(
            "WARNING: DIFFERENT SIZE gold: {} guess: {}".format(
                len(gold_records), len(guess_records)
            )
        )

    # align order
    gold_ids = []
    for gold in gold_records:
        assert str(gold["id"]).strip() not in gold_ids, "Gold IDs should be unique"
        gold_ids.append(str(gold["id"]).strip())

    id2guess_record = {}
    for guess in guess_records:
        assert (
            str(guess["id"]).strip() not in id2guess_record
        ), "Prediction IDs should be unique"
        id2guess_record[str(guess["id"]).strip()] = guess

    guess_records = []
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])
        else:
            raise ValueError("ERROR: no prediction provided for id: {}".format(id))

    return gold_records, guess_records


def evaluate(gold, guess):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = load_data(gold)
    guess_records = load_data(guess)

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    # 2. retrieval performance
    retrieval_results = retrieval_metrics.compute(
        gold_records, guess_records, ks=[1, 5], rank_keys=["wikipedia_id"]
    )
    result["retrieval"] = {
        "Rprec": retrieval_results["Rprec"],
        "recall@5": retrieval_results["recall@5"],
    }

    pp.pprint(result)
    return result



import argparse
import pprint
from collections import defaultdict, OrderedDict


def _remove_duplicates(obj):
    obj_tmp = []
    for o in obj:
        if o not in obj_tmp:
            obj_tmp.append(o)
    return obj_tmp


def _get_ids_list(datapoint, rank_keys, verbose=False):
    # collect all gold ids
    ids_list = []
    for output in datapoint["output"]:
        current_ids_list = []
        if "provenance" in output:
            for provenance in output["provenance"]:
                if any(rank_key not in provenance for rank_key in rank_keys):
                    missing = set(rank_keys) - set(
                        list(provenance.keys())
                    ).intersection(set(rank_keys))
                    if verbose:
                        print(
                            f"WARNING: missing key(s) {missing} in provenance, unable to compute retrieval for those."
                        )
                else:
                    current_ids_list.append(
                        "+".join(
                            [
                                str(provenance[rank_key]).strip()
                                for rank_key in rank_keys
                            ]
                        )
                    )
        ids_list.append(_remove_duplicates(current_ids_list))  # remove duplicates

    # consider only unique ids
    return ids_list


def get_rank(guess_item, gold_item, k, rank_keys, verbose=False):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, "k must be a positive integer grater than 0."

    rank = []
    num_distinct_evidence_sets = 0

    guess_ids = _get_ids_list(guess_item, rank_keys)[0]

    if guess_ids and len(guess_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in gold_item["output"]:
            if "provenance" in output:
                e_set = {
                    "+".join(
                        [
                            str(provenance[rank_key]).strip()
                            for rank_key in rank_keys
                            if rank_key in provenance
                        ]
                    )
                    for provenance in output["provenance"]
                }
                if e_set not in evidence_sets:  # no duplicate evidence set
                    evidence_sets.append(e_set)
                    e_size[len(e_set)] += 1
        num_distinct_evidence_sets = len(evidence_sets)

        # 2. check what's the minimum number of predicted pages needed to get a robust P/R@k
        min_prediction_size = 0
        c = 0
        for size, freq in sorted(e_size.items(), reverse=True):
            for _ in range(freq):
                min_prediction_size += size
                c += 1
                if c == k:
                    break
            if c == k:
                break
        # if the number of evidence sets is smaller than k
        min_prediction_size += k - c

        if verbose and len(guess_ids) < min_prediction_size:
            print(
                f"WARNING: you should provide at least {min_prediction_size} provenance items for a robust recall@{k} computation (you provided {len(guess_ids)} item(s))."
            )

        # 3. rank by gruping pages in each evidence set (each evidence set count as 1),
        # the position in the rank of each evidence set is given by the last page in guess_ids
        # non evidence pages counts as 1
        rank = []
        for guess_id in guess_ids:
            guess_id = str(guess_id).strip()
            found = False
            for idx, e_set in enumerate(evidence_sets):

                e_set_id = f"evidence_set:{idx}"

                if guess_id in e_set:
                    found = True

                    # remove from the rank previous points referring to this evidence set
                    if e_set_id in rank:
                        rank.remove(e_set_id)

                    # remove the guess_id from the evidence set
                    e_set.remove(guess_id)

                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
                    else:
                        # add a point for this partial evidence set
                        rank.append(e_set_id)

            if not found:
                rank.append(False)

    return rank, num_distinct_evidence_sets


# 1. Precision computation
def _precision_at_k(rank, k):

    # precision @ k
    p = rank[:k].count(True) / k

    return p


# 2. Recall computation
def _recall_at_k(rank, num_distinct_evidence_sets, k):

    r = rank[:k].count(True) / num_distinct_evidence_sets

    return r


# 3. Success rate computation
def _success_rate_at_k(rank, k):

    # success rate @ k
    p = int(True in rank[:k])

    return p


# 4. Answer in context computation
def _answer_in_context_at_k(guess_item, gold_item, k):

    answers = get_gold_answers(gold_item)

    if "provenance" in guess_item["output"][0]:
        provenance = guess_item["output"][0]["provenance"]
        for i in range(0, min(k, len(provenance))):
            if "text" in provenance[i]:
                normalized_text = normalize_answer(
                    provenance[i]["text"]
                )
                for a in answers:
                    if normalize_answer(a) in normalized_text:
                        return 1
    return 0


# 5. Answer+entity in context computation
def _answer_and_ent_in_context_at_k(guess_item, gold_item, k):

    answers = get_gold_answers(gold_item)
    titles = get_gold_titles(gold_item)

    if "provenance" in guess_item["output"][0]:
        provenance = guess_item["output"][0]["provenance"]
        for i in range(0, min(k, len(provenance))):
            if "text" in provenance[i]:
                normalized_text = normalize_answer(
                    provenance[i]["text"]
                )
                has_answer = False
                for a in answers:
                    if normalize_answer(a) in normalized_text:
                        has_answer = True
                        break
                if has_answer:
                    for t in titles:
                        if normalize_answer(t) in normalized_text:
                            return 1

    return 0


# 6. Entity in input
def _entity_in_input(gold_item):

    input = normalize_answer(gold_item["input"])
    titles = get_gold_titles(gold_item)

    for t in titles:
        if normalize_answer(t) in input:
            return 1
    return 0


# 7. Entity in context
def _ent_in_context_at_k(guess_item, gold_item, k):

    titles = get_gold_titles(gold_item)

    if "provenance" in guess_item["output"][0]:
        provenance = guess_item["output"][0]["provenance"]
        for i in range(0, min(k, len(provenance))):
            if "text" in provenance[i]:
                normalized_text = normalize_answer(
                    provenance[i]["text"]
                )
  
                for t in titles:
                    if normalize_answer(t) in normalized_text:
                        return 1

    return 0

def _computeRprec(guess_ids, gold_ids):

    R = len(gold_ids)
    num = 0

    for prediction in guess_ids[:R]:
        if str(prediction).strip() in gold_ids:
            num += 1

    Rprec = num / R if R > 0 else 0
    return Rprec


# R-precision https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_486
def rprecision(guess_item, gold_item, rank_keys):
    gold_ids_list = _get_ids_list(gold_item, rank_keys)
    guess_ids = _get_ids_list(guess_item, rank_keys)[0]
    Rprec_vector = []
    for gold_ids in gold_ids_list:
        Rprec = _computeRprec(guess_ids, gold_ids)
        Rprec_vector.append(Rprec)
    return max(Rprec_vector)


def get_ranking_metrics(guess_item, gold_item, ks, rank_keys):

    Rprec = 0.0
    P_at_k = {"precision@{}".format(k): 0 for k in sorted(ks) if k > 0}
    R_at_k = {"recall@{}".format(k): 0 for k in sorted(ks) if k > 1}
    S_at_k = {"success_rate@{}".format(k): 0 for k in sorted(ks) if k > 1}
    A_at_k = {"answer_in_context@{}".format(k): 0 for k in sorted(ks) if k > 0}
    AE_at_k = {"answer_and_ent_in_context@{}".format(k): 0 for k in sorted(ks) if k > 0}
    E_at_k = {"entity_in_context@{}".format(k): 0 for k in sorted(ks) if k > 0}

    assert (
        "output" in guess_item and len(guess_item["output"]) == 1
    ), f"guess should provide exactly one output for {guess_item['id']}"

    Rprec = rprecision(guess_item, gold_item, rank_keys=rank_keys)
    eii = _entity_in_input(gold_item)
    for k in ks:

        # 0. get rank
        rank, num_distinct_evidence_sets = get_rank(
            guess_item, gold_item, k, rank_keys=rank_keys
        )

        if num_distinct_evidence_sets > 0:

            # 1. precision
            P_at_k["precision@{}".format(k)] = _precision_at_k(rank, k)

            # 2. recall
            R_at_k["recall@{}".format(k)] = _recall_at_k(
                rank, num_distinct_evidence_sets, k
            )

            # 3. success rate
            S_at_k["success_rate@{}".format(k)] = _success_rate_at_k(rank, k)

        # 4. answer in context
        A_at_k["answer_in_context@{}".format(k)] = _answer_in_context_at_k(
            guess_item, gold_item, k
        )

        AE_at_k[
            "answer_and_ent_in_context@{}".format(k)
        ] = _answer_and_ent_in_context_at_k(guess_item, gold_item, k)

        E_at_k[
            "entity_in_context@{}".format(k)
        ] = _ent_in_context_at_k(guess_item, gold_item, k)

    return {
        "Rprec": Rprec,
        **P_at_k,
        **R_at_k,
        **S_at_k,
        **A_at_k,
        **AE_at_k,
        **E_at_k,
        "entity_in_input": eii,
    }


def compute(gold_dataset, guess_dataset, ks, rank_keys):

    ks = sorted([int(x) for x in ks])

    result = OrderedDict()
    result["Rprec"] = 0.0
    result["entity_in_input"] = 0.0
    for k in ks:
        if k > 0:
            result["precision@{}".format(k)] = 0.0
            result["answer_in_context@{}".format(k)] = 0.0
            result["answer_and_ent_in_context@{}".format(k)] = 0.0
            result["entity_in_context@{}".format(k)] = 0.0
        if k > 1:
            result["recall@{}".format(k)] = 0.0
            result["success_rate@{}".format(k)] = 0.0

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert (
            str(gold["id"]).strip() == str(guess["id"]).strip()
        ), "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks, rank_keys)
        result["Rprec"] += ranking_metrics["Rprec"]
        result["entity_in_input"] += ranking_metrics["entity_in_input"]
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] += ranking_metrics[
                    "precision@{}".format(k)
                ]
                result["answer_in_context@{}".format(k)] += ranking_metrics[
                    "answer_in_context@{}".format(k)
                ]
                result["answer_and_ent_in_context@{}".format(k)] += ranking_metrics[
                    "answer_and_ent_in_context@{}".format(k)
                ]
                result["entity_in_context@{}".format(k)] += ranking_metrics[
                    "entity_in_context@{}".format(k)
                ]
            if k > 1:
                result["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]
                result["success_rate@{}".format(k)] += ranking_metrics[
                    "success_rate@{}".format(k)
                ]
    if len(guess_dataset) > 0:
        result["Rprec"] /= len(guess_dataset)
        result["entity_in_input"] /= len(guess_dataset)
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] /= len(guess_dataset)
                result["answer_in_context@{}".format(k)] /= len(guess_dataset)
                result["answer_and_ent_in_context@{}".format(k)] /= len(guess_dataset)
                result["entity_in_context@{}".format(k)] /= len(guess_dataset)
            if k > 1:
                result["recall@{}".format(k)] /= len(guess_dataset)
                result["success_rate@{}".format(k)] /= len(guess_dataset)

    return result


def filter_answers(guess):
    new_guess = []
    for x in guess:
        new_o = []
        for o in x["output"]:
            if "provenance" in o:
                new_o.append(o)
        assert (
            len(new_o) == 1
        ), f"guess should provide exactly one output with provenance for {x['id']}"
        x["output"] = new_o
        new_guess.append(x)
    return new_guess


def evaluate(gold, guess, ks, rank_keys):
    pp = pprint.PrettyPrinter(indent=4)

    gold_dataset = load_data(gold)
    guess_dataset = load_data(guess)

    # 0. validate input
    gold_dataset, guess_dataset = validate_input(
        gold_dataset, guess_dataset
    )

    # 1. filter out ground thruth answers
    guess_dataset = filter_answers(guess_dataset)

    # 2. get retrieval metrics
    result = compute(gold_dataset, guess_dataset, ks, rank_keys)

    pp.pprint(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")
    parser.add_argument(
        "--ks",
        type=str,
        required=False,
        default="1,5,10,20",
        help="Comma separated list of positive integers for recall@k and precision@k",
    )
    parser.add_argument(
        "--rank_keys",
        type=str,
        required=False,
        default="title",
        help="Comma separated list of rank keys for recall@k and precision@k",
    )

    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]
    args.rank_keys = [rank_key for rank_key in args.rank_keys.split(",")]

    evaluate(args.gold, args.guess, args.ks, args.rank_keys)