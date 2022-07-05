import argparse
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_named_intro_regex(row):
    # regex match at least 2 names
    # TODO there are still problems with some names (df[df["text"] == ''])
    name_regex = r"((" + "|".join(row["dep_name"].split()) + ").*?){2}"
    party = row["dep_parl_group"] if row["dep_parl_group"] != "PEV" else "Os Verdes"
    named_intro_regex = fr"^[\w\W]*[Sr.ª|Sr.] {name_regex} \({party}\): —"
    return named_intro_regex


def get_intro_regex(row, explicit_found=True):
    named_intro_regex = get_named_intro_regex(row)
    implicit_speaker_regex = fr"^[\w\W]*[Orador|Oradora]: —"
    if explicit_found:
        return f"{named_intro_regex}|{implicit_speaker_regex}"
    else:
        return named_intro_regex


def strip_entries_shorter_than_threshold(entries, threshold):
    return list(filter(lambda x: len(x.split()) > threshold, entries))


def strip_pdf_header(text):
    # example = "'29 DE SETEMBRO DE 2006  19 \nciso uma mudança a montante e é isso que os senhores não abordam: a verdadeira autonomia das esco-\nlas. \nQue poder efectivo é que as escolas portuguesas devem ter?"
    # example2 = "'29 DE SETEMBRO DE 2006 17 escolha é um patamar essencial?'"
    header1 = r"\d+\s+I SÉRIE — NÚMERO \d+"  # "1456                                                    I SÉRIE — NÚMERO 25")
    header2 = r"\d{1,2} DE [A-Z]+ DE \d{4}\s+\d+"  # "28 DE NOVEMBRO DE 2003                1455")
    header_regex = f"{header1}|{header2}"
    text = re.sub(header_regex, "", text)
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw text")
    parser.add_argument(
        "--legs",
        type=str,
        nargs="+",
        default=["X", "XI", "XII", "XIII", "XIV", "XV"],
        choices=["VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"],
        help="legislatures",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/out_with_text.csv",
        help="input file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/out_with_text_processed.csv",
        help="output file path",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_path)
    df["text"] = df["text"].replace(r"^\s*$", np.nan, regex=True)

    print(f"Selected legs: {args.legs}")

    selected_entries_df = df[df.ini_leg.isin(args.legs)]
    left_entries_df = df[~df.ini_leg.isin(args.legs)]
    selected_entries_df = selected_entries_df.reset_index(drop=True)

    no_speech_count = 0
    for row_idx, row in tqdm(
        selected_entries_df.iterrows(), total=len(selected_entries_df)
    ):
        if not row["text"] or isinstance(row["text"], float):
            print(f"Text value is invalid. Skipping entry #{row_idx}")
            continue

        all_but_last_extra = row["text"].split("|||")[:-1]

        explicit_found = False
        lastly_added = False
        speeches_with_name = []
        for i, page_content in enumerate(all_but_last_extra):
            speeches = page_content.split("\n \n")
            speeches = strip_entries_shorter_than_threshold(speeches, 10)
            for j, speech in enumerate(speeches):
                speech = strip_pdf_header(speech)
                # starts with speaker intro
                if re.search(
                    get_intro_regex(row, explicit_found), speech, flags=re.IGNORECASE
                ):
                    speeches_with_name.append(speech)
                    lastly_added = True
                    explicit_found = True
                    continue

                # does not start with speaker intro and previous content was page break
                # ^[\w\W]*[Sr.ª|Sr.] {name_regex}
                # if (not re.search(r"^[\w\W]+ \([\w\W]+\): —", speech, flags=re.IGNORECASE)
                if (
                    not re.search(
                        r"^[\w\W]*[Sr.ª|Sr.] [\w\W]*: —", speech, flags=re.IGNORECASE
                    )
                    and speeches_with_name
                    and lastly_added
                    # and j == 0 # first speech of page
                ):
                    speeches_with_name.append(speech)
                    # lastly_added = False
                    continue

                lastly_added = False

            # last entry is truncated speech
            if (
                i == len(all_but_last_extra) - 1  # last page
                and speeches_with_name
                and lastly_added
            ):
                additional_page_speeches = row["text"].split("|||")[-1]
                additional_page_speeches = additional_page_speeches.split("\n \n")
                first_speech_additional_page = next(
                    x for x in additional_page_speeches if len(x.split()) > 10
                )
                speeches_with_name.append(first_speech_additional_page)

        if len(speeches_with_name) == 0:
            print(f"Could not retrieve speeches for entry #{row_idx}")
            no_speech_count += 1
            selected_entries_df.at[row_idx, "text_process_label"] = 0
            continue

        # join speech fragments
        for i in range(len(speeches_with_name)):
            speech = speeches_with_name[i]
            # remove speaker intro
            speech = re.sub(
                get_intro_regex(row, True), "", speech, flags=re.IGNORECASE
            ).strip()
            # replace linebreaks with spaces
            speech = speech.replace("\n", " ")
            # replace multiple spaces with single space
            speeches_with_name[i] = re.sub("\\s+", " ", speech)

        total_speech = " ".join(speeches_with_name)

        # clean fragmented words because of breakline with hyphen
        # TODO: fix instances where this should not happen (e.g. agradeço'-'lhe)
        clean_speech = []
        speech_words = total_speech.split()
        i = 0
        while i < len(speech_words):
            if speech_words[i].endswith("-"):
                clean_speech.append(speech_words[i][:-1] + speech_words[i + 1])
                i += 2
            else:
                clean_speech.append(speech_words[i])
                i += 1

        clean_speech = " ".join(clean_speech)
        clean_speech = clean_speech.replace("… …", " ")  # \u2026

        selected_entries_df.at[row_idx, "text"] = clean_speech
        selected_entries_df.at[row_idx, "text_process_label"] = 1

    if no_speech_count > 0:
        print(f"Could not retrieve speeches for {no_speech_count} entries.")

    df = pd.concat([selected_entries_df, left_entries_df])
    df["text_process_label"] = df["text_process_label"].fillna(0)

    df.to_csv(args.output_path, index=False)

    #### create three-file structure
    legislature_df = df[["ini_leg", "leg_begin_date", "leg_end_date"]].drop_duplicates()
    initiatives_df = df[
        [
            "ini_leg",
            "ini_num",
            "ini_type",
            "ini_title",
            "ini_session",
            "authors",
            "vot_results",
            "vot_in_favour",
            "vot_against",
            "vot_abstention",
        ]
    ].drop_duplicates()
    interventions_df = df[
        ["ini_num", "dep_id", "dep_name", "dep_parl_group", "text", "vote"]
    ]

    legislature_df.to_csv("data/legislature.csv", index=False)
    initiatives_df.to_csv("data/initiatives.csv", index=False)
    interventions_df.to_csv("data/interventions.csv", index=False)


# %%
