import os
import glob
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
from torch import multiprocessing as mp

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq
# import pyarrow.feather as feather

import pandas as pd
from tqdm.auto import tqdm
from nltk.corpus import wordnet as wn
from lemminflect import getAllInflections

from set_os_env import set_os_env
set_os_env()

### corpora
TRAIN_CORPORA = [
        ("data/semcor.xml.gz", 9e5, "complete_tagging", "paragraph", 4),
        ("data/wngt.xml.gz", 1.8e6, "partial_tagging", "sentence", 32),
        ## ("data/omsti.xml.gz", 2.3e9, "partial_tagging", "sentence", 32),
        ## ("data/trainomatic.xml.gz", 2e9, "partial_tagging", "sentence", 32),
]
TEST_CORPORA = [
        # ("data/semcor.xml", 9e5, "partial_tagging", "paragraph", 4),
        ("data/test/raganato_senseval2.xml.gz", 6.2e3, "partial_tagging", "sentence", 8),
        ("data/test/raganato_senseval3.xml.gz", 6.2e3, "partial_tagging", "sentence", 8),
        ("data/test/raganato_semeval2007.xml.gz", 3.4e3, "partial_tagging", "sentence", 8),
        ("data/test/raganato_semeval2013.xml.gz", 9.0e3, "partial_tagging", "sentence", 8),
        ("data/test/raganato_semeval2015.xml.gz", 2.9e3, "partial_tagging", "sentence", 8),
        ("data/test/raganato_ALL.xml.gz", 2.7e4, "partial_tagging", "sentence", 8),
    ]

TARGET_HIDDEN_LEVEL = 5  # hidden level number to take ctx values from
# HIDDEN_PRECISION = torch.float16  # precision of training examples to save: larger -> dataset file larger
HIDDEN_PRECISION = torch.float32  # precision of training examples to save: larger -> dataset file larger
NUMBER_FAKE_EXAMPLES = 3  # number of examples to generate for unencountered tokens

# _ns - non-sense - too many senses
# _cap - sense limit to use
SENSE_COUNT_LIMITS = {
    'n_ns'  : 80,
    'n_cap' : 64,
    'v_ns' : 128,
    'v_cap' : 96,
    'j_ns'  : 39,
    'j_cap' : 23,
    'r_ns'  : 29,
    'r_cap' : 16
}

MAX_SENSE_COUNT = sum([v for k, v in SENSE_COUNT_LIMITS.items() if 'cap' in k]) + 1  # +1 for default 0-th sense

# dataset paths
BASE_DATASETS_DIR = '/hdd-data/datasets/'  # where parquet will be dumped
BASE_DATASETS_DIR = BASE_DATASETS_DIR if os.path.exists(BASE_DATASETS_DIR) else os.path.join(os.getcwd(), "data")  # or create with os.makedirs
GENERATED_DIR = os.path.join(BASE_DATASETS_DIR, 'generated/')
CACHE_DIR = os.path.join(BASE_DATASETS_DIR, 'hf_cache/')
CHUNK_SIZE = 64 * 1024  # number of examples in a single dataset chunk
MAX_SENSE_EXAMPLES_LIMIT = 64


def main(model_name_or_path, batch_size_override=None):
    token_sense_vocabulary, reverse_token_sense_vocabulary, token_list = build_token_sense_voc_from_corpora(
        model_name_or_path, num_workers=os.cpu_count())
    hf_dataset, short_hf_dataset = build_train_dataset_from_corpora(
        model_name_or_path, reverse_token_sense_vocabulary, batch_size_override)
    print(hf_dataset, short_hf_dataset)
    build_test_dataset_from_corpora(model_name_or_path, num_workers=(2 if torch.cuda.is_available() else os.cpu_count()))

    print("Done. Go home.")


def build_token_sense_voc_from_corpora(model_name_or_path, num_workers=0):
    corpora = TRAIN_CORPORA

    pos_of_interest = ['n', 'v', 'j', 'r']
    # https://wordnet.princeton.edu/documentation/senseidx5wn
    # 1    NOUN    2    VERB    3    ADJECTIVE    4    ADVERB    5    ADJECTIVE SATELLITE
    pos2id = {'1': 'n', '2': 'v', '3': 'j', '4': 'r', '5': 'j'}

    pos_for_voc_name = '_'.join([p for p in pos_of_interest])
    corpora_for_voc_name = '_'.join([c[0].split('/')[-1].split('.')[0] for c in corpora])
    cache_file_path = os.path.join(GENERATED_DIR,
                                   f"token_sense_voc_{model_name_or_path}_{pos_for_voc_name}_{corpora_for_voc_name}.pt")

    if os.path.exists(cache_file_path):
        cached = torch.load(cache_file_path)
        token_sense_vocabulary = cached["token_sense_vocabulary"]
        reverse_token_sense_vocabulary = cached["reverse_token_sense_vocabulary"]
        token_list = cached["token_list"]
        return token_sense_vocabulary, reverse_token_sense_vocabulary, token_list

    worker_args = [(corpus[0], corpus[1], model_name_or_path, pos_of_interest, pos2id) for corpus in corpora]
    if num_workers > 0:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_workers) as pool:
            execution_results = [pool.apply_async(_token_sense_voc_from_corpus_worker, args) for args in worker_args]
            pool.close()
            pool.join()
        all_corpora_pos_tokens = [result.get() for result in execution_results]
    else:
        # # single process for debugging
        all_corpora_pos_tokens = [_token_sense_voc_from_corpus_worker(*args) for args in worker_args]

    tokenizer_vocab = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True).get_vocab()

    # for k in all_pos_tokens.keys():
    #     _show_pandas_gui(all_pos_tokens[k])

    token_sense_vocabulary, reverse_token_sense_vocabulary = _prune_and_cap_token_senses(tokenizer_vocab, pos_of_interest, all_corpora_pos_tokens)

    # put tokenizer vocab into sorted list, need it later
    token_list = [k for k, v in sorted(tokenizer_vocab.items(), key=lambda item: item[1])]

    # cache for future
    torch.save({
        'token_sense_vocabulary': token_sense_vocabulary,
        'reverse_token_sense_vocabulary': reverse_token_sense_vocabulary,
        'token_list': token_list
    }, cache_file_path)

    # show_pandas_gui(token_sense_vocabulary)
    return token_sense_vocabulary, reverse_token_sense_vocabulary, token_list

def _token_sense_voc_from_corpus_worker(corpus, corpus_len, model_name_or_path, pos_of_interest, pos2id):
    wn_cache = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    all_pos_tokens = {}
    for pos in pos_of_interest:
        all_pos_tokens[pos] = defaultdict(set)

    input = gzip.open(corpus, 'r') if corpus.endswith(".gz") else corpus
    events_of_interest = ['start']
    for event, element in tqdm(ET.iterparse(input, events_of_interest),
                               desc=f"Building token sense vocab from {corpus}",
                               total=int(corpus_len) * len(events_of_interest)):
        pos, sense_key, surface_form = None, None, None
        to_tokenize = set()

        if element.tag == 'word' and 'wn30_key' in element.attrib.keys():  # semcor.xml, wngt.xml
            pos = element.attrib['pos'][0].lower()
            sense_key = element.attrib['wn30_key']
            surface_form = element.attrib['surface_form'].split('_')
        elif element.tag == 'sentence' and 'wn30_key' in element.attrib.keys():  # wngt.xml
            sense_key = element.attrib['wn30_key']

        if sense_key is not None:
            if pos is None:
                pos = pos2id[sense_key.split('%')[1][0]]  # sorry...
            if pos not in pos_of_interest:
                continue
            if surface_form is not None:
                to_tokenize.update(surface_form)  # do I need it? really? But then why semcor at all?

            ss = _get_synset(sense_key, wn_cache)
            if ss is not None:
                ss_name = ss.name()

                to_tokenize.update(_get_all_inflections(ss, pos))
                tokens = set(tokenizer.tokenize(" ".join(to_tokenize)))
                for token in tokens:
                    # double size just to be able to analyze later
                    if len(all_pos_tokens[pos][token]) < 2 * SENSE_COUNT_LIMITS[pos + '_ns']:
                        all_pos_tokens[pos][token].add(ss_name)
        element.clear()  # otherwise memory leaks
    return all_pos_tokens

def get_cached_model_vocab(model_name_or_path):
    file_mask = os.path.join(GENERATED_DIR, f"token_sense_voc_{model_name_or_path}_*.pt")
    files = list(glob.glob(file_mask))
    if len(files) > 0:
        cached = torch.load(files[0])  # just first one
        token_sense_vocabulary = cached["token_sense_vocabulary"]
        reverse_token_sense_vocabulary = cached["reverse_token_sense_vocabulary"]
        token_list = cached["token_list"]
        return token_sense_vocabulary, reverse_token_sense_vocabulary, token_list
    else:
        raise FileNotFoundError(os.path.join(os.getcwd(), file_mask))

def _show_token_sense_voc(model_name_or_path):
    token_sense_vocabulary, reverse_token_sense_vocabulary, token_list = get_cached_model_vocab(model_name_or_path)
    _show_pandas_gui(token_sense_vocabulary)
    _show_pandas_gui(reverse_token_sense_vocabulary)


def _prune_and_cap_token_senses(tokenizer_vocab, pos_of_interest, all_corpora_pos_tokens):
    # build final token_sense_vocabulary and reverse_token_sense_vocabulary with sparse map senses

    if not isinstance(all_corpora_pos_tokens, list):
        all_corpora_pos_tokens = [all_corpora_pos_tokens]
    combined_keys = {}
    for pos in pos_of_interest:
        combined_keys[pos] = set()
        [combined_keys[pos].update(corpus[pos].keys()) for corpus in all_corpora_pos_tokens]

    def sense_sort_key(x): return int(x.split('.')[-1])

    token_sense_vocabulary = {}
    reverse_token_sense_vocabulary = {}
    for token in tqdm(tokenizer_vocab, desc="Pruning and capping senses for tokens"):
        senses_map = {}
        reverse_senses_map = {}
        next_sense_id = 1  # skipping default 0-th sense
        for pos in pos_of_interest:
            token_pos_senses = []
            non_sense_limit = SENSE_COUNT_LIMITS[pos + '_ns']
            sense_cap_limit = SENSE_COUNT_LIMITS[pos + '_cap']

            if token in combined_keys[pos]:
                token_pos_senses = set()
                [token_pos_senses.update(corpus[pos][token]) for corpus in all_corpora_pos_tokens]  # set of senses
                # token_pos_senses = all_pos_tokens[pos][token]  # set of senses
                # prune / cap sense number

                if len(token_pos_senses) >= non_sense_limit:
                    token_pos_senses.clear()  # no senses, default sense is added per-token, not per pos, above
                elif len(token_pos_senses) > sense_cap_limit:
                    # sort by sense frequency
                    sorted_token_pos_senses = sorted(token_pos_senses, key=sense_sort_key)
                    # take first sense_cap_limit senses
                    token_pos_senses.clear()
                    # TODO: do I really need order here after capping?
                    token_pos_senses.update(sorted_token_pos_senses[:sense_cap_limit])
            else:
                # token was not encountered in corpora, but not doing anything about it
                ...

            # for sense in token_pos_senses:
            for sense in sorted(token_pos_senses, key=sense_sort_key):
                senses_map[next_sense_id] = sense
                reverse_senses_map[sense] = next_sense_id
                next_sense_id += 1
            next_sense_id += sense_cap_limit -len(token_pos_senses)  # aligning senses for pos

        if len(senses_map) > 0:
            token_sense_vocabulary[token] = senses_map  # map of all substantial senses for each of pos_of_interest
            reverse_token_sense_vocabulary[token] = reverse_senses_map  # map to get sense_id by sense_key for a token

    return token_sense_vocabulary, reverse_token_sense_vocabulary


def build_train_dataset_from_corpora(model_name_or_path, reverse_token_sense_vocabulary, batch_size_override):
    """
        # tokenize paragraph
            # for each token
                # get token_word_idx
                # token_sense.append(word_sense.at(token_word_idx))  # len(list(token)) of senses: 0, -1->UNK?, wn30_key
                # skip -1; zip(token, sense_key) -> list(per_token_senses): [token_sense_voc[token].find(sense_key) -> [0..MAX_SENSE_COUNT]
                # run tokenized through model, get hiddens@layer -> [768d,...]
                # for each token:
                    # skip if special like CLS, SEP, PAD...
                    # dataset[token][token_sense_id].append(768d)  # dataset['run'][41]->[768d, 768d]
        # fix for unencountered tokens:
            # for each unencountered token:
                # 3 (as avg sense count) random values to teach that token's ctx value does not matter and only 0-th sense should be predicted
                # dataset[unencountered_token][0].append(np.random((1, 768), -1, 1)).as_list()...)
        # finally: dataset size == vocab_size, each key has a map of size 1..MAX_SENSE_COUNT with 1-3..5-6-7 ctx_hidden examples

    :param model_name_or_path:
    :param reverse_token_sense_vocabulary:
    :return:
    """

    corpora = TRAIN_CORPORA

    dataset_name = get_dataset_name_for_corpora(model_name_or_path, corpora, type='train')
    if not check_generated_ds_exists(dataset_name):
        dataset = MyDatasetWriter(name=dataset_name, type='train', write_parquet=True, reverse_token_sense_vocabulary=reverse_token_sense_vocabulary)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        wn_cache = {}

        # will be processed sequentially because single train dataset
        # with example limit counter and etc.
        for corpus in corpora:
            worker_args = {
                'corpus': corpus[0],
                'corpus_len': corpus[1],
                'completely_tagged': corpus[2] == "complete_tagging",
                'fragment_mark': corpus[3],  # paragraph or sentence
                'corpus_batch_size': batch_size_override or corpus[4],  # batch size to feed to model
                'dataset': dataset,
                'model_name_or_path': model_name_or_path,
                'tokenizer': tokenizer,
                'model': model,
                'wn_cache': wn_cache
            }
            _build_dataset_from_corpus_worker(**worker_args)
        _add_examples_for_unencountered_tokens(dataset, tokenizer)

    # call it so it is converted from parquet to arrow and cached
    hf_dataset = get_hf_dataset(dataset_name)
    short_hf_dataset = get_hf_dataset(dataset_name, pcent=10)
    return hf_dataset, short_hf_dataset


def build_test_dataset_from_corpora(model_name_or_path, batch_size_override=None, num_workers=0):
    # unlike train dataset where all corpora go in one dataset to
    # keep meaningful 0 sense-ids for unencountered tokens,
    # each test corpora goes into separate dataset

    corpora = TEST_CORPORA

    args_list = []
    for corpus in corpora:
        dataset_name = get_dataset_name_for_corpora(model_name_or_path, [corpus], type='test')[0]
        if check_generated_ds_exists(dataset_name):
            print(f"Test dataset {dataset_name} already exists, skipping.")
            continue
        dataset = MyDatasetWriter(name=dataset_name, type='test', write_parquet=True)

        args = {
            'corpus': corpus[0],
            'corpus_len': corpus[1],
            'completely_tagged': corpus[2] == "complete_tagging",
            'fragment_mark': corpus[3],  # paragraph or sentence
            'corpus_batch_size': batch_size_override or corpus[4],  # batch size to feed to model
            'dataset': dataset,
            'model_name_or_path': model_name_or_path,
            # 'model': AutoModel.from_pretrained(model_name_or_path)
        }
        args_list.append(args)

    if num_workers > 0:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_workers) as pool:
            execution_results = [pool.apply_async(_build_dataset_from_corpus_worker, kwds=args) for args in args_list]
            pool.close()
            pool.join()
        results = [result.get() for result in execution_results]
    else:
        results = [_build_dataset_from_corpus_worker(**worker_args) for worker_args in args_list]

    print(results)


def _build_dataset_from_corpus_worker(corpus, corpus_len, completely_tagged, fragment_mark, corpus_batch_size, dataset, model_name_or_path, tokenizer=None, model=None, wn_cache=None):
    current_batch = []  # list of text fragments
    current_text_fragment = []  # list of words
    sense_name_list = []  # list of word sense key OR 0 for completely-tagged OR -1 for partially-tagged
    test_ex_id_list = []  # list of example ids for test corpora

    wn_cache = wn_cache or {}
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
    model = model or AutoModel.from_pretrained(model_name_or_path)

    input = gzip.open(corpus, 'r') if corpus.endswith(".gz") else corpus
    events_of_interest = ['start', 'end']
    for event, element in tqdm(ET.iterparse(input, events_of_interest),
                               desc=f"Extracting examples from {corpus}",
                               total=int(corpus_len) * len(events_of_interest)):
        if event == 'start' and element.tag == 'sentence' and 'wn30_key' in element.attrib:  # workaround for wngt.xml corpus
            word_list, sense_list = _generate_wngt_sentence_header(element.attrib['wn30_key'], wn_cache)
            if len(word_list) > 0:
                current_text_fragment.extend(word_list)
                sense_name_list.extend(sense_list)
        elif event == 'end':
            if element.tag == 'word':
                words = element.attrib['surface_form'].split('_')
                current_text_fragment.extend(words)

                if 'wn30_key' in element.attrib.keys():
                    ss = _get_synset(element.attrib['wn30_key'], wn_cache)
                    ss_name = ss.name() if ss is not None else -1
                else:
                    ss_name = 0 if completely_tagged else -1
                sense_name_list.extend(
                    [ss_name] * len(words))  # duplicate sense key for each word of multi-word synsets

                ex_id = -1
                if 'id' in element.attrib.keys():
                    ex_id = element.attrib['id']
                test_ex_id_list.extend([ex_id] * len(words))
            elif element.tag == fragment_mark:
                current_batch.append((current_text_fragment, sense_name_list,
                                      test_ex_id_list))  # put current text fragment and sense_name list into batch
                if len(current_batch) == corpus_batch_size:
                    _process_batch(dataset, model, tokenizer, current_batch)
                    current_batch = []
                current_text_fragment = []
                sense_name_list = []
                test_ex_id_list = []
            elif element.tag == 'document':
                if len(current_batch) > 0:
                    _process_batch(dataset, model, tokenizer, current_batch)  # process final batch for document
            elif element.tag == 'corpus':
                if dataset.type == 'test':
                    dataset.dump_chunk(is_last=True)
            # inside 'end' event, otherwise all element data is cleared after 'start'
            element.clear()
    return f"dataset builder for {corpus} have finished"


def _process_batch(dataset, model, tokenizer, fragments_batch):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    fragment_words_list, fragment_sense_list, fragment_test_ex_id_list = zip(*fragments_batch)
    encoded_input = tokenizer(fragment_words_list, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input, output_attentions=True, output_hidden_states=True)
    model_hiddens = model_output.hidden_states[TARGET_HIDDEN_LEVEL].detach().cpu()  # tensor batch x len x model-d
    zipped_data = zip(encoded_input.encodings, fragment_sense_list, fragment_test_ex_id_list)
    if dataset.type == 'train':
        _parse_batch_for_train(zipped_data, model_hiddens, dataset)
    elif dataset.type == 'test':
        _parse_batch_for_test(zipped_data, model_hiddens, dataset)
    dataset.dump_chunk()
    
    
def _parse_batch_for_train(zipped_data, model_hiddens, dataset):
    batch_idx = 0
    for encoding, sense_list, _ in zipped_data:
        for idx, token in enumerate(encoding.tokens):
            if encoding.special_tokens_mask[idx] == 0:  # don't look at special tokens
                token_sense_id = None
                sense_id = sense_list[encoding.word_ids[idx]]  # token_word_id = encoding.word_ids[idx]
                if sense_id == -1:  # skip tokens for untagged words in partially tagged corpora -> only 0 and wn30_key... should remain
                    continue
                elif sense_id == 0:  # default / non-sense sense
                    token_sense_id = 0
                else:  # wn30_key...
                    token_sense_id = dataset.reverse_token_sense_vocabulary[token][sense_id] \
                        if (token in dataset.reverse_token_sense_vocabulary and sense_id in dataset.reverse_token_sense_vocabulary[token]) else 0

                token_id = encoding.ids[idx]
                ctx_tensor = model_hiddens[batch_idx, idx].to(HIDDEN_PRECISION)  # should be 768-d value for token
                train_dataset_item = _build_train_dataset_item(token, token_id, token_sense_id, ctx_tensor)
                dataset.add_item(train_dataset_item)
        batch_idx += 1
    
def _parse_batch_for_test(zipped_data, model_hiddens, dataset):
    batch_idx = 0
    for encoding, sense_list, ex_id_list in zipped_data:
        skip_till_idx = -1
        for idx, token in enumerate(encoding.tokens):
            if idx >= skip_till_idx and encoding.special_tokens_mask[idx] == 0:  # don't look at special tokens
                # for test corpora it should be -1 or wn_key
                sense_id = sense_list[encoding.word_ids[idx]]  # token_word_id = encoding.word_ids[idx]
                if sense_id == -1:  # not test example word
                    continue
                else:
                    ex_id = ex_id_list[encoding.word_ids[idx]]
                    # count how many next tokens have save word_id / in how many tokens a word was tokenized
                    word_end_token_idx = idx + sum([1 for x in encoding.word_ids[idx:] if x == encoding.word_ids[idx]])
                    word_tokens = encoding.tokens[idx:word_end_token_idx]
                    word_token_ids = encoding.ids[idx:word_end_token_idx]
                    word_token_ctxs = model_hiddens[batch_idx, idx:word_end_token_idx].to(HIDDEN_PRECISION)
                    test_dataset_item = _build_test_datset_item(ex_id, sense_id, word_tokens, word_token_ids, word_token_ctxs)
                    dataset.add_item(test_dataset_item)
                    skip_till_idx = word_end_token_idx
        batch_idx += 1
    

def _add_examples_for_unencountered_tokens(dataset, tokenizer):
    number_of_fake_examples = NUMBER_FAKE_EXAMPLES
    hidden_dim = dataset.get_item()[-1].shape[0] - 1  # -1 for subtracting token_id from model hidden
    tokenizer_vocab = tokenizer.get_vocab()
    counter = 0
    for token in tokenizer_vocab:
        if token not in dataset.encountered_tokens:
            counter += 1
            for _ in range(number_of_fake_examples):
                random_value = torch.FloatTensor(hidden_dim,).uniform_(-1, 1).to(HIDDEN_PRECISION)
                train_dataset_item = _build_train_dataset_item(token, tokenizer_vocab[token], 0, random_value)
                dataset.add_item(train_dataset_item)
    print(f"Added {counter}*{number_of_fake_examples} ({counter*number_of_fake_examples}) 0-th sense fake examples for unencountered tokens")
    dataset.dump_chunk(is_last=True)


def _build_train_dataset_item(token, token_id, token_sense_id, ctx_value):
    # return (token, token_id, token_sense_id, ctx_hidden_value)
    merged_together = torch.hstack((torch.tensor(token_id, dtype=torch.int32), ctx_value))
    return [token, token_sense_id, merged_together.numpy()]

def _build_test_datset_item(ex_id, ss_id, token_lst, token_id_lst, ctx_lst):
    ctxs = [ctx_lst[i].numpy() for i in range(ctx_lst.shape[0])]
    return [ex_id, ss_id, token_lst, token_id_lst, ctxs]


def _get_synset(sense_key, wn_cache):
    if sense_key in wn_cache:
        return wn_cache[sense_key]

    # fixing key especially for adj/%3 -> they are not rerieved from current version of WN 3.0 as they are in corpora
    # https://wordnet.princeton.edu/documentation/senseidx5wn
    # fixed_key = ":".join(sense_key.split(';')[0].split(":")[0:3]) + "::"
    fixed_key = sense_key.split(';')[0]
    # if sense_key != fixed_key:
    #     print("Fixed wn key: ", sense_key, ' -> ', fixed_key)

    ss = None
    try:
        ss = wn.synset_from_sense_key(fixed_key)
    except Exception as e:
        if "%3" in fixed_key:
            # smth strange with semcor.xml, semcor in nltk and wordnet in terms of %3 and %5
            try_key = fixed_key.replace('%3', '%5')
            ss = wn.synset_from_sense_key(try_key)
            # print("Replacing %3 with %5 helped for: ", fixed_key)
        else:
            print(f"While retrieving from WN using key {fixed_key} got an exception: {e}")

    if ss:
        wn_cache[sense_key] = ss

    return ss


def _get_all_inflections(wn_synset, tgt_pos):
    """
    ??? https://stackoverflow.com/questions/67342461/how-to-generate-all-derived-terms-out-of-a-root-or-lemma-word-in-english-using-s
    """

    all_forms = set()
    for lemma in wn_synset.lemmas():
        lemma_words = lemma.name().split("_")
        all_forms.update(lemma_words)
        for word in lemma_words:
            all_inflections = getAllInflections(word)
            for pos, value in all_inflections.items():
                if pos[0].lower() == tgt_pos:
                    all_forms.update(set(value))

    return all_forms


def _generate_wngt_sentence_header(sense_key, wn_cache):
    word_list = []
    sense_list = []

    ss = _get_synset(sense_key, wn_cache)
    if ss is not None:
        ss_name = ss.name()
        # TODO: do I need training example for each possible form of each possible lemma? maybe the other time...
        for lemma in ss.lemmas():
            lemma_words = lemma.name().split("_")
            word_list.extend(lemma_words)
            sense_list.extend([ss_name] * len(lemma_words))
            word_list.append(",")
            sense_list.append(-1)
        # remove last comma / sense_name for it
        word_list.pop()
        sense_list.pop()
        # TODO: add 'to' for verbs
        word_list.append("means")
        sense_list.append(-1)  # to not use this 'means' in any training

    return word_list, sense_list

# cached dataset related stuff
def get_dataset_name_for_corpora(model_name, corpora, type='train'):
    name = ''
    if type=='train':
        # multiple corpora -> single dataset
        name = f"{type}_ds_{model_name}_{'_'.join([(c[0].split('/')[-1]).split('.')[0] for c in corpora])}"
    elif type=='test':
        # each corpus in a single dataset
        name = [f"{type}_ds_{model_name}_{(corpus[0].split('/')[-1]).split('.')[0]}" for corpus in corpora]
    return name

def check_generated_ds_exists(dataset_name):
    return os.path.exists(os.path.join(GENERATED_DIR, dataset_name))

def get_hf_dataset(ds_name, keep_in_memory=False, pcent=100, seed=42):
    data_dir = os.path.join(GENERATED_DIR, ds_name)
    params = {
        'name': ds_name,
        'version': '0.0.1',
        'data_dir': data_dir,
        'data_files': None,
        'description': 'Temporary cache for training',
        #
        'cache_dir': CACHE_DIR,
    }

    ds = load_dataset('parquet', keep_in_memory=keep_in_memory, **params)
    if 'train' in ds_name:
        ds.set_format(type='torch', columns=['token_sense_id', 'merged-token_id-ctx_value'])
        if pcent < 100:
            # use first train/test split to cut requested protion of dataset and second split for actual train/test splits
            ds_subset = ds['train'].train_test_split(train_size=pcent / 100.0, seed=seed)
            splits = ds_subset['train'].train_test_split(test_size=0.1, seed=seed)
        else:
            splits = ds['train'].train_test_split(test_size=0.1)
    elif 'test' in ds_name:
        ds.set_format(type='torch', columns=['ex_id', 'ss_id', 'token_lst', 'token_id_lst', 'ctx_lst'])
        splits = ds

    return splits


def get_tensor_dataset(model_name_or_path, corpus_filename):
    tmp_cache_filename = os.path.join(GENERATED_DIR, "last_ds_cache.pt")
    if os.path.exists(tmp_cache_filename):
        dataset = torch.load(tmp_cache_filename,  map_location=torch.device('cpu'))
        print("Last cached dataset was loaded.")
        return dataset

    ds_name = f"ds_{model_name_or_path}_{corpus_filename.split('.')[0]}"
    ds_dir = os.path.join(GENERATED_DIR, ds_name+"_pt")

    src_data = []
    tgt_data = []

    def get_chunk_filename():
        return os.path.join(ds_dir, f"{ds_name}_chunk_{chunk_idx}.pt")

    chunk_idx = 0
    while os.path.exists(get_chunk_filename()):
        chunk = torch.load(get_chunk_filename(), map_location=torch.device('cpu'))
        for example in tqdm(chunk, desc=f"Reading {get_chunk_filename()}..."):
            token, token_sense_id, merged_token_id_ctx_value = example
            # glue together token_id and ctx_hidden, will be separated later in model's forward
            # src_data.append(torch.hstack((torch.tensor(token_id, dtype=torch.float32), torch.tensor(ctx_hidden))))  # moved into dataset creation
            src_data.append(torch.tensor(merged_token_id_ctx_value, dtype=torch.float32))
            tgt_data.append(torch.tensor(token_sense_id, dtype=torch.long))
        print(f"Chunk {get_chunk_filename()} processed.")
        chunk_idx += 1
    src = torch.stack(src_data)
    tgt = torch.stack(tgt_data)

    data_len = tgt.shape[0]
    # no sorting, leave order as it happened
    data_sort_idx = torch.arange(data_len)

    rand_perm = torch.randperm(data_len)  # random order to sample val/test sets randomly from sorted data
    val_size = test_size = data_len // 10  # 10% for val and test sets
    val_set_idx = data_sort_idx[rand_perm[:val_size]]
    test_set_idx = data_sort_idx[rand_perm[val_size:val_size + test_size]]
    train_set_idx = data_sort_idx[rand_perm[val_size + test_size:]]

    dataset = {
        'train': (src[train_set_idx], tgt[train_set_idx]),
        'val': (src[val_set_idx], tgt[val_set_idx]),
        'test': (src[test_set_idx], tgt[test_set_idx])
    }

    torch.save(dataset, tmp_cache_filename)

    return dataset

class MyDatasetWriter:
    def __init__(self, name=None, type='train', write_parquet=True, reverse_token_sense_vocabulary=None):
        self.name = name
        self.type = type
        self.output_columns = ['token', 'token_sense_id', 'merged-token_id-ctx_value'] if self.type == 'train' else \
            ['ex_id', 'ss_id', 'token_lst', 'token_id_lst', 'ctx_lst']
        self.write_parquet = write_parquet
        self.data = []
        self.encountered_tokens = set()
        self.reverse_token_sense_vocabulary = reverse_token_sense_vocabulary
        self.chunk_idx = 0
        self.token_sense_example_cnt = defaultdict(int)
        self.skipped_sense_example_cnt = defaultdict(int)


    def add_item(self, item):
        if self.type == 'test':
            self.data.append(item)
        elif self.type == 'train':
            token, token_sense_id, _ = item
            self.encountered_tokens.add(token)
            key = token + '__' + str(token_sense_id)
            count = self.token_sense_example_cnt[key]
            if token_sense_id != 0 or count < MAX_SENSE_EXAMPLES_LIMIT:
                self.data.append(item)
                count += 1
                self.token_sense_example_cnt[key] = count
            else:
                count = self.skipped_sense_example_cnt[token_sense_id]
                count += 1
                self.skipped_sense_example_cnt[token_sense_id] = count

    def get_item(self, idx=0):
        return self.data[idx]

    def dump_chunk(self, is_last=False):
        chunk_size = CHUNK_SIZE
        if len(self.data) < chunk_size and not is_last:
            return

        # TODO: change to queue?
        while len(self.data) > chunk_size or (is_last and (len(self.data) > 0)):
            chunk_filename = self.name + f"_chunk_{self.chunk_idx}" + (".parquet" if self.write_parquet else ".pt")
            chunk_dir = os.path.join(GENERATED_DIR, self.name + ("" if self.write_parquet else "_pt"))
            if not os.path.exists(chunk_dir):
                os.makedirs(chunk_dir)
            chunk_path = os.path.join(chunk_dir, chunk_filename)
            print(f"{'Final dumping ' if is_last else 'Dumping '} {chunk_path} ...")

            if self.write_parquet:
                # https://stackoverflow.com/questions/47113813/using-pyarrow-how-do-you-append-to-parquet-file
                df = pd.DataFrame(data=self.data[0:chunk_size],
                                  columns=self.output_columns)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, chunk_path, compression="zstd")
                # feather.write_feather(df, chunk_path, compression='zstd')
            else:
                torch.save(self.data[0:chunk_size], chunk_path)

            del self.data[0:chunk_size]
            self.chunk_idx += 1
            print("Dumped.")
            if is_last and self.type == 'train':
                print(f"Skipped examples by sense_id: ", self.skipped_sense_example_cnt)


def _setup_requirements():
    print("!pip install -U nltk lemminflect transformers")

    import nltk
    nltk.download('wordnet')


def _show_pandas_gui(dict=None):
    import pandas as pd
    from pandasgui import show

    df = pd.DataFrame().from_dict(data={k: len(v) for k, v in dict.items()}, orient='index')
    show(df)


if __name__ == '__main__':
    main('distilbert-base-uncased')
    # _show_token_sense_voc('distilbert-base-uncased')
    # main('bert-base-uncased')

