# Paraphrase generator

Paraphrase generation with round-trip machine translation, for the Human Language Engineering course in UPC-BarcelonaTech Master in Artificial Intelligence (HLE-MAI).

## Description

The task of automatically generating paraphrases is relevant for a variety of reasons. First, they are interesting per se as a proof task for Natural Language Processing (NLP) models and linguistics. Second, rephrasing a sentence is a common task for all kind of writers. Last, but not least, it could be used for data augmentation in NLP, which is more challenging than in vision (since we cannot apply spatial transformations such as rotation or cropping). In this work, we explore the well-known technique of paraphrasing via round-trip machine translation from different perspectives. Specifically, we study the effect of using different families of languages, speech recognition and synthesis systems, and multiple translation cycles. Apart from manually inspecting the generated paraphrases for an intrinsic qualitative analysis, we evaluate them with different intrinsic quantitative metrics, and use them as data augmentation in a sentence classification task, as an extrinsic evaluation. We propose a new intrinsic evaluation metric, Jaccard-Embedding Factor, which accounts for both string/word-level distance (which must be maximized, to obtain real paraprhases instead of just copies), and the cosine similarity of the sentence embeddings obtained with SentenceBERT (https://github.com/UKPLab/sentence-transformers).

## Paraphrasing methods:

- Vanilla round-trip: English -> Language A -> English
- Intermediate: English -> Language A -> Language B -> English
- Ncycles: English -> Language A -> English

## Translators:

Note that we also use inter-modal transformations (speech).

- Fairseq pre-trained translators (German, Tamil,...): https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
- Marian pre-trained translators (Icelandic, Russian,...): https://huggingface.co/transformers/model_doc/marian.html
- Text-To-Speech (TTS) with Tacotron: https://github.com/ttaoREtw/Tacotron-pytorch
- Speech recognition with PyTorch Silero models: https://pytorch.org/hub/snakers4_silero-models_stt/

## Results

### Intrinsic evaluation

We defined our metric, Jaccard-Embedding Factor, as: jaccard_distance * cos_sim_sentencebert. Good paraphrases maximize this metric. Apart from this metric, we collect other metrics.

- Intrinsic quantitative metrics for the vanilla round-trip machine translation paraphrasers:

| extsc{Model}    | \textsc{bleu} | \textsc{emb. cos. sim.} | \textsc{norm. edit dist.} | \textsc{jaccard} | \textsc{jaccard-emb.-factor} |
|-----------------|---------------|-------------------------|---------------------------|------------------|------------------------------|
| German          | 0.77 (0.17)   | 0.88 (0.11)             | 6.18 (18.87)              | 0.38 (0.22)      | 0.32 (0.17)                  |
| Icelandic       | 0.50 (0.20)   | 0.70 (0.19)             | 12.37 (28.90)             | 0.64 (0.17)      | 0.43 (0.13)                  |
| Russian         | 0.58 (0.20)   | 0.78 (0.15)             | 12.36 (30.87)             | 0.59 (0.18)      | 0.45 (0.13)                  |
| \textbf{Speech} | 0.65 (0.20)   | 0.71 (0.16)             | 9.97 (5.50)               | 0.67 (0.21)      | \textbf{0.46 (0.16)}         |

- Intrinsic quantitative metrics comparing paraphrasing methods

| extsc{Model}    | \textsc{bleu} | \textsc{emb. cos. sim.} | \textsc{norm. edit dist.} | \textsc{jaccard} | \textsc{jaccard-emb.-factor} |
|-----------------|---------------|-------------------------|---------------------------|------------------|------------------------------|
| \textbf{Speech} | 0.65 (0.20)   | 0.71 (0.16)             | 9.97 (5.50)               | 0.67 (0.21)      | \textbf{0.46 (0.16)}         |
| Speech N=2      | 0.62 (0.20)   | 0.66 (0.17)             | 10.47 (5.54)              | 0.69 (0.21)      | 0.44 (0.15)                  |


### Extrinsic evaluation

For the extrinsic evaluation, we use the generated paraphrases as data augmentation in a sentence classification task (see extrinsic_evaluation) directory. We use the sentence classifier and data from: https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch

- Test accuracy in the extrinsic evaluation (the sentence classification task from a Kaggle competition based on tweets we said) for the vanilla round-trip machine translation paraphasers:

| extsc{Model}    | \textsc{test accuracy} |
|-----------------|------------------------|
| Baseline        | 0.735                  |
| German          | 0.661                  |
| Icelandic       | 0.678                  |
| Russian         | 0.686                  |
| \textbf{Speech} | \textbf{0.748}         |

- Test accuracy in the extrinsic evaluation for different paraphrasing methods):

| extsc{Model}    | \textsc{test accuracy} |
|-----------------|------------------------|
| Baseline        | 0.735                  |
| \textbf{Speech} | \textbf{0.748}         |
| Speech N=2      | 0.734                  |

## Setup

Download dependencies:

```
bash setup.sh
```


## Usage

Activate the virtual environment with `source venv/bin/activate`. Then:

- Generate paraphrases:

```
python paraphrase.py --method {dummy, roundtrip, intermediate, ncycles} --translators [SPACE_SEPARATED_TRANSLATORS_LIST] --input input/tweets.csv
```

The output of each run is stored in a new directory within the `output/`directory. Examples of translator names one can use:
`
--translators silero_asr_en tacotron_pytorch
--translators marian-en-ru marian-ru-en
--translators marian-en-de marian-de-en
--translators marian-en-is marian-is-en
`

- Evaluate paraphrases:

```
python evaluate.py output/[PARAPHRASES_OUTPUT_DIRECTORY]
```

## Directory tree
