from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

STOPWORD_SET = set(stopwords.words('english'))


class EntityMatcher:
    def __init__(self, entities, remove_stopwords=True):
        self.entities = entities

        self.split_entities = [
            tuple(entity.lower().split('_')) for entity in self.entities]
        self.entities_map = dict(zip(self.split_entities, self.entities))

        # Remove entities which are stopwords or composed entirely of stopwords
        if remove_stopwords:
            self.split_entities = [
                split_entity for split_entity in self.split_entities if
                not all([entity in STOPWORD_SET for entity in split_entity])]

        self.split_entities = set(self.split_entities)
        self.max_entity_len = len(max(self.split_entities, key=len))

    def match_entities(self, text, remove_stopwords=False):
        text = text.lower()
        text = word_tokenize(text)

        if remove_stopwords:
            text = [word for word in text if word not in STOPWORD_SET]

        num_matched = 0
        matched = [False for _ in range(len(text))]
        matched_entities = set()
        for n in range(len(text), 0, -1):
            text_n_grams = ngrams(text, n)
            for i, n_gram in enumerate(text_n_grams):
                if all([matched[j] for j in range(i, i + n)]):
                    continue

                if n_gram in self.split_entities:
                    matched_entities.add(self.entities_map[n_gram])
                    for j in range(i, i + n):
                        num_matched += 1
                        matched[j] = True

            if num_matched >= len(matched):
                break

        return list(matched_entities)
