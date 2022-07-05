import argparse
import ast
import math
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
from tqdm import tqdm


def actual_page(row, last=True):
    pad = 0 if last else 1
    factor = 1.13
    actual_page = int(row["pages"].split("-")[last]) - int(row["doc_first_page"])
    return math.floor(actual_page * factor) + pad


def get_page_content(page, older_leg=True):
    if older_leg:
        x0 = 0  # Distance of left side of character from left side of page.
        x1 = 0.5  # Distance of right side of character from left side of page.
        y0 = 0  # Distance of bottom of character from bottom of page.
        y1 = 1  # Distance of top of character from bottom of page.

        width = page.width
        height = page.height

        # Crop pages
        left_bbox = (
            x0 * float(width),
            y0 * float(height),
            x1 * float(width),
            y1 * float(height),
        )
        page_crop = page.crop(bbox=left_bbox)
        left_text = page_crop.extract_text()

        left_bbox = (
            0.5 * float(width),
            y0 * float(height),
            1 * float(width),
            y1 * float(height),
        )
        page_crop = page.crop(bbox=left_bbox)
        right_text = page_crop.extract_text()
        page_content = "\n".join([left_text, right_text])
        return page_content
    else:
        return page.extract_text()


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    results = pool.map(func, df_split)
    df_slices, no_spch_cnt_slices = list(zip(*results))
    df = pd.concat(df_slices)
    no_speech_count = sum(no_spch_cnt_slices)
    pool.close()
    pool.join()
    return df, no_speech_count


def add_raw_text(df):
    no_speech_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_idx = row.name
        older_leg = row["ini_leg"] in ["VII", "VIII", "IX"]

        if older_leg:
            first = actual_page(row, last=False)
            last = actual_page(row, last=True)
        else:
            # clean noisy page numbers
            first, last = map(
                lambda x: int(re.findall(r"\d+", x)[0]), row["pages"].split("-")
            )

        file_path = Path("data/pdf_minutes", row["ini_leg"], row["pdf_file_path"],)

        try:
            with pdfplumber.open(file_path) as pdf:
                for j, page in enumerate(pdf.pages):
                    page_content = get_page_content(page, older_leg)
                    if (
                        first - 1 <= j <= last
                    ):  # add one to `last` to minimize truncated speeches
                        if "text" not in df or pd.isnull(df.at[row_idx, "text"]):
                            df.at[row_idx, "text"] = page_content
                        else:
                            df.at[row_idx, "text"] = (
                                df.at[row_idx, "text"] + "|||" + page_content
                            )
        except IOError:
            print(f"File not accessible at: {file_path}")
            no_speech_count += 1

    return df, no_speech_count


def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val


def parse_args():
    parser = argparse.ArgumentParser(description="Add raw text from PDFs")
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
        # default="data/initial_corpus_meta.csv",
        help="input file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/out_with_text.csv",
        help="output file path",
    )
    parser.add_argument(
        "--resume_from", type=str, help="file path to resume from",
    )
    parser.add_argument(
        "--log_path", type=str, default="general.log", help="log file path",
    )
    args = parser.parse_args()

    if args.input_path and args.resume_from:
        raise ValueError(
            "--resume_from cannot be specified along " "--input_path and --output_path"
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.resume_from:
        input_file_path = args.resume_from
    else:
        input_file_path = args.input_path
    print(f"Input file path: {input_file_path}")
    df = pd.read_csv(input_file_path)
    df["pages"] = df["pages"].apply(literal_return)
    df = df.explode("pages")

    print(f"Selected legs: {args.legs}")

    selected_entries_df = df[df.ini_leg.isin(args.legs)]
    left_entries_df = df[~df.ini_leg.isin(args.legs)]
    selected_entries_df = selected_entries_df.reset_index(drop=True)

    selected_entries_df, no_speech_count = parallelize_dataframe(
        selected_entries_df, add_raw_text
    )

    print(f"Could not access files for {no_speech_count} entries.")

    df = pd.concat([selected_entries_df, left_entries_df])

    df.to_csv(args.output_path, index=False)
