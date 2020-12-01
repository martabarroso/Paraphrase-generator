from unittest import TestCase

from paraphraser.paraphraser import Paraphraser


class TestParaphraser(TestCase):
    def test_roundtrip_paraphraser(self):
        expected = {'en': [['When they go on field trips, they let parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on excursions, they let their parents go and help.',
                            'Here we can only take a bus.']],
                    'de': [['Wenn sie auf Exkursionen gehen, lassen sie die Eltern gehen und helfen.',
                            'Hier können wir nur einen Bus nehmen.']]}

        paraphraser = Paraphraser.build('roundtrip', ['fair-wmt19-en-de', 'fair-wmt19-de-en'])
        n = 1
        sentences = ['When they go on field trips, they let parents go and help.', 'Here we can only take one bus.']
        paraphrases = paraphraser.n_paraphrase_sentences(sentences, n, language_list=['en', 'de', 'en'])

        self.assertDictEqual(expected, paraphrases)

    def test_two_cycle_paraphraser(self):
        expected = {'en': [['When they go on field trips, they let parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on excursions, they let their parents go and help.',
                            'Here we can only take a bus.'],
                           ['When they go on trips, they let their parents go and help.',
                            'Here we can only go by bus.']],
                    'de': [['Wenn sie auf Exkursionen gehen, lassen sie die Eltern gehen und helfen.',
                            'Hier können wir nur einen Bus nehmen.'],
                           ['Wenn sie Ausflüge machen, lassen sie ihre Eltern gehen und helfen.',
                            'Hier können wir nur mit dem Bus fahren.']]}

        paraphraser = Paraphraser.build('roundtrip', ['fair-wmt19-en-de', 'fair-wmt19-de-en'])
        n = 1
        sentences = ['When they go on field trips, they let parents go and help.', 'Here we can only take one bus.']
        paraphrases = paraphraser.n_paraphrase_sentences(sentences, n, language_list=['en', 'de', 'en', 'de', 'en'])
        self.assertDictEqual(expected, paraphrases)











