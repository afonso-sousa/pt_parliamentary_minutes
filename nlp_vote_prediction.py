# %%
# In the scope of the [DARGMINTS](https://web.fe.up.pt/~dargmints/) project, we have collected
# a corpus of debates of the Portuguese Parliament regarding specific parliamentary initiatives
# that are subject to voting. For each debate, we have discourses of each participating deputy,
# his/her party, and how the party has voted on that specific diploma.

# The goal of this project is to build a classifier for predicting how a given party (or person)
# will vote, based on the respective discourse.

import argparse
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

tqdm.pandas()
nltk.download("stopwords")


def get_stopwords():
    pt_stopwords = stopwords.words("portuguese")
    dataset_specific_stopwords = [
        "sr.",
        "sr",
        "sr.ª",
        "sr.as",
        "srs.",
        "srs",
        "secretário",
        "secretários",
        "secretária",
        "proposta",
        "governo",
        "presidente",
        "estado",
        "deputado",
        "deputados",
        "ministro",
        "ministros",
        "partido",
        "partidos",
    ]
    return pt_stopwords + dataset_specific_stopwords


def entry_cleanup(r: pd.Series) -> pd.Series:
    # Remove non word characters
    r = re.sub("\W+", " ", r)
    # Text to lower case
    r = r.lower()
    # Remove stopwords
    r = [
        word
        for word in word_tokenize(r, language="portuguese")
        if word not in (get_stopwords())
    ]
    r = " ".join(r)

    return r


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test", "attention_vis"],
        nargs="?",
        help="whether to train or test the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["nb", "lr", "bert"],
        nargs="?",
        help="model architecture",
    )
    args = parser.parse_args()

    return args


# %%
# if __name__ == "__main__":

# args = parse_args()

from argparse import Namespace

args = Namespace(model = "bert", mode="test")


normalized_dataset_path = Path("data/normalized_dataset.csv")
if args.model in ["nb", "lr"]:
    if normalized_dataset_path.is_file():
        print("Normalized dataset found. Loading dataset...")
        dataset = pd.read_csv("data/normalized_dataset.csv")
    else:
        print("Normalized dataset not found. Loading original dataset...")
        print("Importing dataset...")
        dataset = pd.read_csv("data/parliamentary_minutes.csv")
        dataset = dataset[~dataset.isnull().any(axis=1)]
        dataset = dataset.reset_index(drop=True)

        print("Normalizing dataset...")
        dataset["processed_text"] = dataset["text"].progress_apply(
            lambda r: entry_cleanup(r)
        )
        dataset.to_csv("data/normalized_dataset.csv", index=False)

    X = dataset["processed_text"]
    y = dataset["vote"]
else:
    print("Importing dataset...")
    dataset = pd.read_csv("data/out_with_text_processed.csv")
    dataset = dataset[~dataset.isnull().any(axis=1)]
    dataset = dataset.reset_index(drop=True)

    X = dataset[["ini_num", "dep_parl_group", "text"]]
    # X = dataset["text"]
    y = dataset["vote"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

print("\nLabel distribution in the training set:")
print(y_train.value_counts(normalize=True))

print("\nLabel distribution in the test set:")
print(y_test.value_counts(normalize=True))

# %%
if args.model == "nb":
    print("Starting Naive Bayes training and eval")
    max_features = 10_000
    selected_features = 3_000
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Select most relevant 3000 features
    selector = SelectKBest(chi2, k=selected_features)
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)

    clf = MultinomialNB()
    clf.fit(X_train_selected, y_train.ravel())
    X_test_features = vectorizer.transform(X_test)
    X_test_selected = selector.transform(X_test_features)
    y_pred = clf.predict(X_test_selected)

    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    print(f"F1: {f1_score(y_test, y_pred, average='macro')}")
elif args.model == "lr":
    print("Starting Logistic Regression training and eval")
    word_embeddings = KeyedVectors.load_word2vec_format("cbow_s50.txt")

    def document_vector(r):
        """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
        r = [
            word
            for word in word_tokenize(r, language="portuguese")
            if word in word_embeddings.index_to_key
        ]

        return np.mean(word_embeddings[r], axis=0)

    X_train_embedded = [
        document_vector(speech)
        for _, speech in tqdm(
            X_train.items(),
            desc="Creating train document vector",
            total=len(X_train),
        )
    ]
    X_test_embedded = [
        document_vector(speech)
        for _, speech in tqdm(
            X_test.items(), desc="Creating test document vector", total=len(X_test)
        )
    ]

    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train_embedded, y_train.ravel())
    y_pred = clf.predict(X_test_embedded)

    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    print(f"F1: {f1_score(y_test, y_pred, average='macro')}")
elif args.model == "bert":
    if args.mode == "train":
        print("\nStarting DistilBERT training and eval")
        X_train = X_train["text"]
        
        model_name = "distilbert-base-multilingual-cased"

        y_train = y_train.replace(
            {"vot_in_favour": 0, "vot_against": 1, "vot_abstention": 2,}
        )  # rename labels to int values
        y_test = y_test.replace(
            {"vot_in_favour": 0, "vot_against": 1, "vot_abstention": 2,}
        )  # rename labels to int values

        train_df = pd.concat([X_train, y_train], axis=1)
        train_df = train_df.rename(columns={"processed_text": "text", "vote": "label"})

        test_df = pd.concat([X_test, y_test], axis=1)
        test_df = test_df.rename(columns={"processed_text": "text", "vote": "label"})

        train_test_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(sample):
            return tokenizer(sample["text"], padding="max_length", truncation=True)

        encoded_dataset = train_test_dataset.map(preprocess_function, batched=True)
        encoded_dataset.set_format(
            type="torch", columns=["input_ids", "label", "attention_mask"]
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )

        training_args = TrainingArguments(
            output_dir="./model_results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            save_strategy="epoch",
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.predict(test_dataset=encoded_dataset["test"])

        device = next(model.parameters()).device

        y_pred = []
        for p in encoded_dataset["test"]["text"]:
            ti = tokenizer(p, return_tensors="pt", truncation=True)
            out = model(**ti.to(device))
            pred = torch.argmax(out.logits).detach().cpu().numpy()
            y_pred.append(pred)

        y_test = encoded_dataset["test"]["label"]

        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
        print(f"F1: {f1_score(y_test, y_pred, average='macro')}")
    elif args.mode == "test":
        print("\nStarting DistilBERT testing")
        def disambiguated_mode(x):
            mode = pd.Series.mode(x)
            if len(mode) == 3:
                return 2 # "vot_abstention"
            if set(mode.values) == set([0, 1]):
                return 2 # "vot_abstention"
            if set(mode.values) == set([0, 2]):
                return 0 # "vot_in_favour"
            if set(mode.values) == set([1, 2]):
                return 1 # "vot_against"
            return pd.Series.mode(x).values[0]

        model_name = "distilbert-base-multilingual-cased"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            "./model_results/checkpoint-924/"
        )

        def preprocess_function(sample):
            return tokenizer(sample["text"], padding="max_length", truncation=True)

        y_test = y_test.replace(
            {"vot_in_favour": 0, "vot_against": 1, "vot_abstention": 2,}
        )  # rename labels to int values

        test_df = pd.concat([X_test, y_test], axis=1)
        test_df = test_df.rename(columns={"processed_text": "text", "vote": "label"})

        test_dataset = DatasetDict({"test": Dataset.from_pandas(test_df)})

        encoded_dataset = test_dataset.map(preprocess_function, batched=True)
        encoded_dataset.set_format(
            type="torch", columns=["input_ids", "label", "attention_mask"]
        )

        device = next(model.parameters()).device
        y_pred = []
        for p in encoded_dataset["test"]["text"]:
            ti = tokenizer(p, return_tensors="pt", truncation=True)
            out = model(**ti.to(device))
            pred = torch.argmax(out.logits).detach().cpu().item()
            y_pred.append(pred)

        y_test = encoded_dataset["test"]["label"]

        print("Simple Testing\n")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
        print(f"F1: {f1_score(y_test, y_pred, average='macro')}")

        test_df = test_df.reset_index(drop=True)
        test_df["preds"] = pd.Series(y_pred, dtype=int)

        agg_labels = test_df.groupby(["ini_num", "dep_parl_group"])["label"].agg(
            lambda x: disambiguated_mode(x)
        ).values

        agg_preds = test_df.groupby(["ini_num", "dep_parl_group"])["preds"].agg(
            lambda x: disambiguated_mode(x)
        ).values    

        print("Testing with aggregated parties\n")
        print("Confusion Matrix:")
        print(confusion_matrix(agg_labels, agg_preds))
        print(f"Accuracy: {accuracy_score(agg_labels, agg_preds)}")
        print(f"Precision: {precision_score(agg_labels, agg_preds, average='macro')}")
        print(f"Recall: {recall_score(agg_labels, agg_preds, average='macro')}")
        print(f"F1: {f1_score(agg_labels, agg_preds, average='macro')}")
    elif args.mode == "attention_vis":
        print("\nStarting attention weight visualization")
        
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        import matplotlib.ticker as ticker
        import numpy as np
        from nltk.tokenize import word_tokenize

        model_name = "distilbert-base-multilingual-cased"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            "./model_results/checkpoint-924/"
        )

        # Visualize attention
        def visualize_attention(input_text, out_attentions, layer=0, head=4):
            attentions = out_attentions[layer].squeeze(0)[head].detach().cpu()
            # Set up figure with colorbar
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(attentions.numpy(), cmap='hot')
            fig.colorbar(cax)

            tokens = word_tokenize(input_text, language="portuguese")

            # Set up axes
            # ax.set_xticks(ax.get_xticks().tolist())
            ax.set_xticklabels(["[CLS]"] + tokens + ["[SEP]"], rotation=90)
            # ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels(["[CLS]"] + tokens + ["[SEP]"])

            # Show label at every tick
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

            # plt.show()
            img_path = f'attention_{layer}_{head}.png'
            print(f"Saving plot in {img_path}")
            plt.savefig(img_path)

        max_length = 10
        head = 10
        layer = 3
        
        device = next(model.parameters()).device

        # input_text = encoded_dataset["test"]["text"][0]
        # input_text = " ".join(word_tokenize(input_text, language="portuguese")[14:14 + max_length])
        input_text = "O que " + \
            "está em causa não foi imposto pelo Governo da República, o que está em " + \
            "causa não foi definido pelo Governo da República e o que está em causa " + \
            "não vai ser concretizado pelo Governo da República."
        ti = tokenizer(input_text, return_tensors="pt", truncation=True)
        out = model(**ti.to(device), output_attentions=True)
        visualize_attention(input_text, out.attentions, layer, head)
    else:
        raise ValueError(
            "Mode not implemented."
        )

# %%
