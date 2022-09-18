# Dual Paraphrase Model for Unsupervised Semantic Parsing

Some terms:

 - **NL** (Natural Language): _article published in 2004_
 - **CF** (Canonical Form): _article whose publication date is 2004_
 - **LF** (Logical Form): `( call SW.listValue ( call SW.filter ( call SW.getProperty ( call SW.singleton en.article  ) ( string !type  )  ) ( string publication_date  ) ( string =  ) ( date 2004 -1 -1  )  )  )`
 - **NL2CF** (Natural Language to Canonical Form Model)
 - **CF2NL** (Canonical Form to Natural Language Model)

----

## Oracle Semantic Parsing

Use natural language sentences as inputs, logical forms as outputs. Actually, this is traditional semantic parser trained on `(nl, lf)` pairs.

----

## Naive Semantic Parsing

Trained on `(cf, lf)` pairs, pick the best model on dev dataset (also `(cf, lf)` pairs), and test on test dataset (this time `(nl, lf)` pairs). This is actually a lower bound.

----

## Pipeline Semantic Parsing

Firstly, use labeled `(nl, cf)` as training data, train a paraphase model. Then evaluate on a naive semantic parser trained on `(cf, lf)` pairs. This pipeline seperates the oracle semantic parsing process into two phases: paraphrase and naive semantic parsing. This is actually an upper bound.

----

## Dual Paraphrase Learning Semantic Parsing

Train a naive semantic parser on `(cf, lf)` pairs in advance. Then train a dual paraphrase model on unlabeled `cf` and `lf` corpus. To pick the best result, since there is no labeled dev dataset(unsupervised settings), fix a maximum epoch number or save the model with maximum reward on dev dataset. When evaluate on test dataset, use `NL2CF` model to first change input `nl` to `cf`, then evaulate on the pretrained naive semantic parser.