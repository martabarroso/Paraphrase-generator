# Paraphrase generator with round-trip machine translation

Paraphrase generation with round-trip machine translation, for the Human Language Engineering course in UPC-BarcelonaTech Master in Artificial Intelligence (HLE-MAI).

## Authors

- Jordi Armengol-Estapé
- Marta Barroso

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

The paraphrases generated with the Speech system seem to be useful as data augmenation. Notice how our proposed intrinsic metric, Jaccard-Embedding Factor, seems to be related with the actual performance in the extrinsic evaluation.

### Intrinsic evaluation

We defined our metric, Jaccard-Embedding Factor, as: jaccard_distance * cos_sim_sentencebert. Good paraphrases maximize this metric. Apart from this metric, we collect other metrics.

- Intrinsic quantitative metrics for the vanilla round-trip machine translation paraphrasers:

| MODEL    | BLEU | EMB. COS. SIM. | NORM. EDIT DIST. | JACCARD DIST. | JACCARD-EMB FACTOR |
|-----------------|---------------|-------------------------|---------------------------|------------------|------------------------------|
| German          | 0.77 (0.17)   | 0.88 (0.11)             | 6.18 (18.87)              | 0.38 (0.22)      | 0.32 (0.17)                  |
| Icelandic       | 0.50 (0.20)   | 0.70 (0.19)             | 12.37 (28.90)             | 0.64 (0.17)      | 0.43 (0.13)                  |
| Russian         | 0.58 (0.20)   | 0.78 (0.15)             | 12.36 (30.87)             | 0.59 (0.18)      | 0.45 (0.13)                  |
| **Speech** | 0.65 (0.20)   | 0.71 (0.16)             | 9.97 (5.50)               | 0.67 (0.21)      | **0.46 (0.16)**         |

- Intrinsic quantitative metrics comparing paraphrasing methods

| MODEL    | BLEU | EMB. COS. SIM. | NORM. EDIT DIST. | JACCARD DIST. | JACCARD-EMB FACTOR |
|-----------------|---------------|-------------------------|---------------------------|------------------|------------------------------|
| **Speech** | 0.65 (0.20)   | 0.71 (0.16)             | 9.97 (5.50)               | 0.67 (0.21)      | **0.46 (0.16)**        |
| Speech N=2      | 0.62 (0.20)   | 0.66 (0.17)             | 10.47 (5.54)              | 0.69 (0.21)      | 0.44 (0.15)                  |


### Extrinsic evaluation

For the extrinsic evaluation, we use the generated paraphrases as data augmentation in a sentence classification task (see extrinsic_evaluation) directory. We use the sentence classifier and data from: https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch

- Test accuracy in the extrinsic evaluation (the sentence classification task from a Kaggle competition based on tweets we said) for the vanilla round-trip machine translation paraphasers:

| MODEL  | TEST ACCURACY |
|-----------------|------------------------|
| Baseline        | 0.735                  |
| German          | 0.661                  |
| Icelandic       | 0.678                  |
| Russian         | 0.686                  |
| **Speech** | **0.748**         |

- Test accuracy in the extrinsic evaluation for different paraphrasing methods):

| MODEL  | TEST ACCURACY |
|-----------------|------------------------|
| Baseline        | 0.735                  |
| **Speech** | **0.748**         |
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

The project contains the following files and directories:

-   `input/`: contains the dataset `tweets.csv`.

-   `output/`: contains different foldersswith the results of
    each experiment. The name of the folder describes the translation
    method used, the name and the languages of the translators, the\tex
    dataset used, the timestamp and the commit of the repository. Each
    of these folders contains:

    -   `args.json`: contains information about the translation method
        used, the translator names and the path of the dataset.

    -   `eval-<timestamp><commit>.json`: contains the results for the
        intrinsic and the extrinsic evaluation. For each sentence, the
        intrinsic evaluation displays the paraphrase and the result of
        the metrics: `bleu_score`, `normalized_original_sentence`,
        `normalized_ paraphrase`, `embedding_cosine_similarity`,
        `edit_distance`, `normalized_edit_distance`, `jaccard` and
        `jaccard_embedding_factor`. At the end, it also shows the
        description (the total number of paraphrases, the mean, the
        standard deviation, the minimum value, the values of the
        percentiles and the maximum value) of each metric.

    -   `eval-<timestamp><commit>.log`: contains the accuracy and the
        loss values for each epoch and the final accuracy of the
        sentence classification task.

    -   `paraphrase.log`: log file of the paraphrase generation. In
        particular it contains information about the time that the
        translation takes for each cycle (i.e the time to translate from
        English to another language and vice versa).

    -   `paraphrases.json`: list of lists where in each position appears
        the original sentence and its paraphrases. In addition, this
        folder contains a .json
        (`paraphrases_four_sources_2021 -01-07-1735.json`) with the
        combination of all the paraphrases generated using different
        languages and converting the text to speech and vice versa.

-   `paraphraser/`: contains the classes for generating and
    evaluating paraphrases.

-   `extrinsic_evaluation/`: contains the classes that implements the
    sentence classification.

-   `tacroton_pytorch/`: contains the implementation of the Google’s
    Tacotron TTS system with PyTorch. This directory is not included
    directly, but downloaded with the `get-third-party.sh` script.

-   `evaluate.py`: main program that evaluates the paraphrases of a
    given folder or directly a .json file.

-   `paraphrase.py`: main program that generates paraphrases given a
    translator method, the names of the translators and the
    input dataset. As a result generates a folder with the results
    located at `output`.

-   `run_extrinsic_baseline.py`: executes the extrinsic evaluation
    baseline (without data augmentation) with `tweet.csv`.

-   `build_report.py`: Utility for generating Latex tables and randomly
    sampling paraphrases for the qualitative analysis.

-   `get-third-party.sh`: download the implementation of Tacrotron TTS
    from the github repository
    `https://github.com/ttaoREtw/Tacotron-pytorch`.

-   `setup.sh`: defines the setup of the project creating the vitual
    environment, installing the requirements and downloading the
    implementation of Tacroton TTS.

-   `requirements.txt`: includes all the libraries used in the project.

