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

    def test_roundtrip_paraphraser_II(self):
        expected = {'en': [['When they go on field trips, they let parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on business trips, they let their parents go and help.',
                            'Here we can only take one bus.']],
                    'ru': [['Когда они выезжают в командировки, они отпускают родителей и помогают.',
                            'Здесь мы можем проехать только на одном автобусе.']]}
        paraphraser = Paraphraser.build('roundtrip', ['fair-wmt19-en-ru', 'fair-wmt19-ru-en'])
        n = 1
        sentences = ['When they go on field trips, they let parents go and help.', 'Here we can only take one bus.']
        paraphrases = paraphraser.n_paraphrase_sentences(sentences, n, language_list=['en', 'ru', 'en'])
        self.assertDictEqual(expected, paraphrases)

    def test_two_cycle_paraphraser(self):
        expected = {'en': [['When they go on field trips, they let parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on business trips, they let their parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on business trips, they let their parents go and help.',
                            'Here we can only take a bus.']],
                    'ru': [['Когда они выезжают в командировки, они отпускают родителей и помогают.',
                            'Здесь мы можем проехать только на одном автобусе.']],
                    'de': [['Wenn sie auf Geschäftsreisen gehen, lassen sie ihre Eltern gehen und helfen.',
                            'Hier können wir nur einen Bus nehmen.']]}

        paraphraser = Paraphraser.build('roundtrip', ['fair-wmt19-en-ru', 'fair-wmt19-ru-en', 'fair-wmt19-en-de',
                                                      'fair-wmt19-de-en'])
        n = 1
        sentences = ['When they go on field trips, they let parents go and help.', 'Here we can only take one bus.']
        paraphrases = paraphraser.n_paraphrase_sentences(sentences, n, language_list=['en', 'ru', 'en', 'de', 'en'])
        self.assertListEqual(expected, paraphrases)










