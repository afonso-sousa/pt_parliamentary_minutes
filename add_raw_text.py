from pathlib import Path

import pandas as pd
import pdftotext


def actual_page(row, last=True):
    return int(row["pages"].split("-")[last]) - int(row["doc_first_page"]) + 1


def create_vote_from_vote_lists(df):
    # explode all three vote columns
    exp_df = (
        df.explode("vot_in_favour").explode("vot_against").explode("vot_abstention")
    )
    # compare labels with votes to find matches
    exp_df = exp_df.eq(exp_df.pop("dep_parl_group"), axis=0)
    # remove all False rows and get the matches in each row
    exp_df = exp_df[exp_df.any(1)].idxmax(1)
    # remove duplicated indices
    df["vote"] = exp_df[~exp_df.index.duplicated()]


if __name__ == "__main__":
    df = pd.read_pickle("data/initial_corpus_meta.pkl")
    df = df.explode("pages")
    df = df.reset_index(drop=True)

    for row_idx, row in df.iterrows():
        first = actual_page(row, last=False)
        last = actual_page(row, last=True)

        file_path = Path(
            "data/parliament/atas_parlamentares_pdfs",
            row["ini_leg"],
            row["pdf_file_path"],
        )
        try:
            with open(file_path, "rb") as f:
                pdf = pdftotext.PDF(f)
                pages_list = []
                for j, page in enumerate(pdf):
                    if (
                        first - 1 <= j <= last
                    ):  # add one to `last` to minimize truncated speeches
                        if "text" not in df:
                            df.at[row_idx, "text"] = page
                        elif pd.isnull(df.at[row_idx, "text"]):
                            df.at[row_idx, "text"] = page
                        else:
                            df.at[row_idx, "text"] = (
                                df.at[row_idx, "text"] + "|||" + page
                            )

            print(f"Entry #{row_idx + 1} - Done successfully")
        except IOError:
            print(f"Entry #{row_idx + 1} - File not accessible")

    # replace cells with empty spaces (namely in text
    # when it cannot be extracted)
    df = df.replace(r"^s*$", float("NaN"), regex=True)
    # removes before-mentioned records
    df = df[~df.isnull().any(axis=1)]

    # process votes
    create_vote_from_vote_lists(df)

    df.to_csv("data/out_with_text.csv", index=False)

