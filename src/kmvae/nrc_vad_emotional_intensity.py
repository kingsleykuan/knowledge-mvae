import numpy as np

NRC_VAD_DEFAULT_VALUE = 0.5
NRC_VAD_DEFAULT = \
    (NRC_VAD_DEFAULT_VALUE, NRC_VAD_DEFAULT_VALUE, NRC_VAD_DEFAULT_VALUE)


class NRCVADEmotionalIntensity:
    def __init__(self, nrc_valence_path, nrc_arousal_path, nrc_dominance_path):
        self.valence = self._read_values(nrc_valence_path)
        self.arousal = self._read_values(nrc_arousal_path)
        self.dominance = self._read_values(nrc_dominance_path)

        self.words = sorted(self.valence.keys())

        self.vad = {
            word: (
                self.valence[word],
                self.arousal[word],
                self.dominance[word],
            )
            for word in self.words}

        self.emotional_intensity, self.default_emotional_intensity = \
            self._calculate_emotional_intensity()

    def get_vad(self, word):
        return self.vad.get(word, NRC_VAD_DEFAULT)

    def get_valence(self, word):
        return self.valence.get(word, NRC_VAD_DEFAULT_VALUE)

    def get_arousal(self, word):
        return self.arousal.get(word, NRC_VAD_DEFAULT_VALUE)

    def get_dominance(self, word):
        return self.dominance.get(word, NRC_VAD_DEFAULT_VALUE)

    def get_emotional_intensity(self, word):
        return self.emotional_intensity.get(
            word, self.default_emotional_intensity)

    def add_synonyms(self, word_synonyms):
        synonym_words = {}
        for word, synonym in word_synonyms:
            if synonym not in synonym_words:
                synonym_words[synonym] = []
            synonym_words[synonym].append(word)

        # If synonym is associated with multiple words, find the average scores
        for synonym, words in synonym_words.items():
            if synonym in self.vad:
                continue

            self.vad[synonym] = float(np.mean(
                [self.vad[word] for word in words]))

            self.valence[synonym] = float(np.mean(
                [self.valence[word] for word in words]))
            self.arousal[synonym] = float(np.mean(
                [self.arousal[word] for word in words]))
            self.dominance[synonym] = float(np.mean(
                [self.dominance[word] for word in words]))

            self.emotional_intensity[synonym] = float(np.mean(
                [self.emotional_intensity[word] for word in words]))

            self.words.append(synonym)
        self.words.sort()

    @staticmethod
    def _read_values(path):
        with open(path) as file:
            lines = file.read().splitlines()
        values = [line.split('\t') for line in lines]
        values = {word.lower(): float(value) for (word, value) in values}
        return values

    def _calculate_emotional_intensity(self):
        emotional_intensity = [
            (self.get_valence(word) - 0.5, self.get_arousal(word) / 2.0)
            for word in self.words]
        # Append default emotional intensity
        emotional_intensity.append(
            (self.get_valence('') - 0.5, self.get_arousal('') / 2.0))
        emotional_intensity = np.asarray(emotional_intensity)

        emotional_intensity = np.linalg.norm(
            emotional_intensity, ord=2, axis=-1)

        min = np.min(emotional_intensity)
        max = np.max(emotional_intensity)
        emotional_intensity = (emotional_intensity - min) / (max - min)

        default_emotional_intensity = emotional_intensity[-1]
        emotional_intensity = emotional_intensity[:-1]

        emotional_intensity = {
            word: float(value)
            for (word, value) in zip(self.words, emotional_intensity)}

        return emotional_intensity, default_emotional_intensity
