# Unsupervised Dual Paraphrasing for Two-stage Semantic Parsing

This is the project containing source code for the paper [Unsupervised Dual Paraphrasing for Two-stage Semantic Parsing](https://arxiv.org/pdf/2005.13485.pdf) in ACL 2020 main conference.

If you find it useful, please cite our work (apologize for the delayed release).

    @inproceedings{cao-etal-2020-unsupervised-dual,
        title = "Unsupervised Dual Paraphrasing for Two-stage Semantic Parsing",
        author = "Cao, Ruisheng  and
        Zhu, Su  and
        Yang, Chenyu  and
        Liu, Chen  and
        Ma, Rao  and
        Zhao, Yanbin  and
        Chen, Lu  and
        Yu, Kai",
        booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
        year = "2020",
        publisher = "Association for Computational Linguistics",
    }

----

Some common terms used in this repository:

 - **nl**: natural language, e.g., _article published in 2004_
 - **cf**: canonical form, utterance generated from grammar rules, e.g., _article whose publication date is 2004_
 - **lf**: logical form, auto-generated paired semantic representation of `cf`, e.g., `( call SW.listValue ( call SW.filter ( call SW.getProperty ( call SW.singleton en.article  ) ( string !type  )  ) ( string publication_date  ) ( string =  ) ( date 2004 -1 -1  )  )  )`
 - **nl2cf**: natural language to canonical form paraphrase model
 - **cf2nl**: canonical form to natural language paraphrase model
 - **cf2lf**: canonical form to logical form _naive_ semantic parser
 - **dataset**: dataset name, in this repository, it can be chosen from `['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork', 'geo']`

## Environment setup

1. Create conda environment and install dependencies:

        conda create -n semparse python=3.7
        conda activate semparse
        pip3 install -r requirements.txt

2. Download third-party `evaluator`/`lib` and pretrained models:
  - Notice that, if the download path of model `GoogleNews-vectors-negative300.bin.gz` is not available, you can download from this [link](https://drive.google.com/file/d/1c0yxK7qtGpgSDVrsh2ruBhR8O4HsEQFu/view?usp=sharing)

        bash ./pull_dependency.sh

3. After downloading all the dependencies, the working repository should have the following directory:

        - data
            - geo: processed dataset files of dataset GeoGranno
            - geo_granno: raw dataset files of dataset GeoGranno
            - overnight: dataset files of dataset OVERNIGHT
            - paraphrase: paraphrases of dataset OVERNIGHT, generated with tool sempre
        - evaluator: dependency downloaded from third-party
        - lib: dependency downloaded from third-party
        - models: all torch modules used in this work
        - pretrained_models: downloaded pre-trained models, include GloVe, GoogleNews word vectors, ELMo and BERT models
        - run: bash running scripts which invoke python programs in scripts
        - scripts: python main programs of different experiments
        - utils: all utility functions

----

## One-stage Semantic Parsing

The semantic parser aims to directly convert the input `nl` into the target `lf`. We consider different baselines depending on whether the annotated `(nl, lf)` pairs are available:

1. **Supervised** settings: the semantic parser is directly trained on `(nl, lf)` pairs. `labeled` denotes the ratio~(`float`) of labeled samples used, e.g., `0.1`.

        ./run/run_one_stage_semantic_parsing.sh [dataset] [labeled]

2. **Unsupervised** settings: the semantic parser is trained on `(cf, lf)` pairs, while evaluated on `(nl, lf)` pairs. Parameter `embed` can be chosen from `['glove', 'elmo', 'bert']`.

        ./run/run_pretrained_embed_semantic_parsing.sh [dataset] [embed]

3. **Unsupervised** pseudo labeling settings: for each unlabeled `nl`, choose the most similar `lf` from the entire `lf` set based on the minimum WMD between `nl` and each `cf`. Then the parser is trained on pseudo labeled `(nl, lf)` pairs.

        ./run/run_one_stage_wmd_samples.sh [dataset]

4. **Unsupervised** multi-tasking settings: the semantic parser is trained on `(cf, lf)` pairs, plus the utterance-level denoising auto-encoder task which converts unlabeled noisy `nl` into its original version. The encoder is shared, while two separate decoder, one for `lf` generation, another for `nl` recovery.

        ./run/run_one_stage_multitask_dae.sh [dataset]


----

## Two-stage Semantic Parsing

The entire semantic parser includes two parts: a _paraphrase model_ and a _naive semantic parser_. The `nl2cf` paraphrase model firstly paraphrases the `nl` into the corresponding `cf`, then the naive semantic parser translates the `cf` into the target `lf`. Notice that `(cf, lf)` pairs are available from the synchronous grammar and can be used to train an off-the-shelf naive semantic parser:

    ./run/run_naive_semantic_parsing.sh [dataset]

The pre-trained downstream parser can be loaded via the argument `--read_nsp_model_path directory_to_model` afterwards.

Next, we experiment in different settings depending on whether the annotated `(nl, cf)` pairs are available.

1. **Supervised** settings: the paraphrase model is trained on labeled `(nl, cf)` pairs.

        ./run/run_two_stage_semantic_parsing.sh [dataset] [labeled]

2. **Unsupervised** pseudo labeling settings: for each unlabeled `nl`, choose the most similar `cf` from the entire `cf` set based on the minimum WMD. Then the `nl2cf` paraphrase model is trained on pseudo labeled `(nl, cf)` pairs.

        ./run/run_two_stage_wmd_samples.sh [dataset]

3. **Unsupervised** multi-tasking settings: we perform two dual utterance-level denoising auto-encoder~(`dae`) tasks, which aims to convert the noisy `nl` or noisy `cf` into the clean version. The encoder is shared for `nl` and `cf`, while a separate decoder for each type of utterance.
  - Notice that, it is also a preliminary task to warmup the dual paraphrase model in cycle learning phase.
  - Default noisy channels include drop, addition and 2-gram shuffling, which can be altered via the argument `--noise_type xxx` in the running script.

        ./run/run_two_stage_multitask_dae.sh [dataset]

4. **Unsupervised/Seimi-supervised** cycle learning settings: based on the pre-trained dual paraphrase model~(`nl2cf` and `cf2nl`) in the two-stage multi-tasking DAE experiment, we apply two additional self-supervised tasks in the cycle learning phase, namely dual back-translation~(`dbt`) and dual reinforcement learning~(`drl`), to further improve the final performance.

    Some auxiliary models, namely two language models~(for `nl` and `cf` respectively) and a text style classifier, need to be pre-trained in order to calculate the fluency~(`flu`) and style~(`sty`) rewards during cycle learning.

        ./run/run_language_model.sh [dataset]
        ./run/run_text_style_classification.sh [dataset]

    By specifying the model directories for dual paraphrase model~(`--read_pdp_model_path xxx`), naive semantic parser~(`--read_nsp_model_path xxx`), language model~(`--read_language_model xxx`) and text style classifier~(`--read_tsc_model_path xxx`), the unsupervised dual paraphrasing cycle can starts:
      - `labeled=0` -> unsupervised setting ; `labeled>0` -> semi-supervised settings
      - the training scheme during cycle learning can be altered via the argument `--train_scheme xxx`
      - noisy channels for `dae` can be altered via the argument `--noise_types xxx` if the `train_scheme` contains `dae`
      - reward types during `drl` can be altered via the argument `--reward_type xxx` if the `train_scheme` contains `drl`

            ./run/run_cycle_learning.sh [dataset] [labeled]

All experiments above use the `torch.device("cuda:0")` by default, which can be changed to other index by changing the argument `--deviceId x` (x=-1 -> cpu, otherwise GPU index).

----

## Acknowledgement

We would like to thank all authors with their pioneer work that provides the datasets and inspires this work.

1. [Building a Semantic Parser Overnight](https://aclanthology.org/P15-1129.pdf)

2. [Don’t paraphrase, detect! Rapid and Effective Data Collection for Semantic Parsing](https://aclanthology.org/D19-1394.pdf)

3. [Semantic Parsing via Paraphrasing](https://aclanthology.org/P14-1133.pdf)