import json
import random
import spacy
from typing import Tuple, Dict, List
from collections import defaultdict

from datasets import load_dataset, IterableDataset
from transformers import (
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

MAX_TURN = 5
HIT_REWARD = 10
MISS_REWARD = -1

class ChatEnv:

    def __init__(self, configs) -> None:

        self.device = configs.device
        self.step_count = 0

        # simulator
        self.sim_model = BlenderbotForConditionalGeneration.from_pretrained(configs.sim_pretrained_model).to(self.device)
        self.sim_tokenizer = BlenderbotTokenizer.from_pretrained(configs.sim_pretrained_model)
        self.max_len = configs.max_len

        # dataset
        self.dataset = self.__prepare_dataset(configs.dataset_split)

        # hit rate calculation
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.keywords = self.__lemmatize(configs.keywords_file)


    def step(self, action: str = None, reset: bool = False) -> Tuple:

        # encode initial sentence / bot's sentence with sim-tokenizer
        if reset:
            self.step_count = 0
            initial_sentence = self.dataset[random.randint(0, len(self.dataset)-1)]['context']
            inputs = self.sim_tokenizer(
                [initial_sentence],
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt',
            ).to(self.device)
        else:
            inputs = self.sim_tokenizer(
                [action],
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt',
            ).to(self.device)

        # generate sim-model's output(greedy)
        sim_ids = self.sim_model.generate(**inputs)

        # decode sim-model's reply with sim-tokenizer
        sim_sentence = self.sim_tokenizer.batch_decode(sim_ids, skip_special_tokens=True)[0]

        # check if simulator's reply hits
        self.step_count += 1
        reward, done = None, None
        if not reset and self.__check_hit(sim_sentence):
            reward = HIT_REWARD
            done = True
        else:
            reward = MISS_REWARD
            done = False if self.step_count < MAX_TURN else True

        return sim_sentence, reward, done


    def __prepare_dataset(self, split: str) -> IterableDataset:
        
        def preprocess(example):
            example["personas"] = [f"your persona: {p}" for p in example["personas"]]
            example["context"] = "\n".join(
                example["personas"]
                + (["additional_context"] if example["additional_context"] else [])
                + example["previous_utterance"]
            )
            return example

        dataset = load_dataset('blended_skill_talk', split=split)
        dataset = dataset.map(
            preprocess,
            remove_columns=[
                'free_messages',
                'guided_messages',
                'suggestions',
                'personas',
                'additional_context',
                'previous_utterance',
            ],
        )
        return dataset


    def __lemmatize(self, keywords_file: str) -> Dict[str, List]:

        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords = json.load(f)

        # lemmatize words in keywords
        for key, val in keywords.items():
            # separate words by its length (one, others)
            one_lemma = []
            multi_lemma = []
            for word in val:
                split = [token.lemma_ for token in self.nlp(word)]
                if len(split) >= 2:
                    multi_lemma.append(" ".join(split))
                else:
                    one_lemma.append(split[0])
                keywords[key] = [one_lemma, multi_lemma]
        
        return keywords


    def __check_hit(self, sentence: str) -> bool:
        
        lemma_utterance = [token.lemma_ for token in self.nlp(sentence)]
        service_hits = defaultdict(int)
        intersection = set()

        for key, (one, multi) in self.keywords.items():
            intersection = set(one) & set(lemma_utterance)
            # check whether the word, the length is bigger than 2, is in the utterance
            for m in multi:
                unsplit_utterance = " ".join(lemma_utterance)
                if m in unsplit_utterance:
                    intersection.add(m)
            service_hits[key] += len(intersection)

        return len(intersection) > 0