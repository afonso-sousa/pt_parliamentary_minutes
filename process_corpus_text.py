import re
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/out_with_text.csv")

    no_speech_count = 0
    for row_idx, row in df.iterrows():
        all_but_last_extra = row["text"].split("|||")[:-1]
        all_but_last_extra = "\n\n".join(all_but_last_extra)
        
        # regex match at least 2 names
        # TODO there are still problems with some names (df[df["text"] == ''])
        name_regex = r"(("+"|".join(row["dep_name"].split())+").*?){2}"
        party = row["dep_parl_group"] if row["dep_parl_group"] != 'PEV' else 'Os Verdes'
        
        speeches = all_but_last_extra.split("\n\n")
        speeches_with_name = []
        break_page = False

        named_intro_regex = fr"^[\w\W]*[Sr.ª|Sr.] {name_regex} \({party}\): —"
        implicit_speaker_regex = fr"^[\w\W]*[Orador|Oradora]: —"
        intro_regex = f"{named_intro_regex}|{implicit_speaker_regex}"
        for i, speech in enumerate(speeches):
            # starts with speaker intro
            if re.search(intro_regex, speech, flags=re.IGNORECASE) and len(speech.split()) > 10:
                speeches_with_name.append(speech)
            # does not start with speaker intro
            elif speeches_with_name and speeches_with_name[-1].endswith("\x0c") and len(speech.split()) > 10:
                speeches_with_name.append(speech)
                # break_page = True
            
            # last entry is truncated speech
            if i == len(speeches) - 1 and speeches_with_name and speeches_with_name[-1].endswith("\x0c"):
                additional_page_speeches = row["text"].split("|||")[-1]
                additional_page_speeches = additional_page_speeches.split("\n\n")
                first_speech_additional_page = next(x for x in additional_page_speeches if len(x.split()) > 10)
                speeches_with_name.append(first_speech_additional_page)

        if len(speeches_with_name) == 0:
            print(f"No speeches for entry #{row_idx}")
            no_speech_count += 1
            
        # join speech fragments
        for i in range(len(speeches_with_name)):
            speech = speeches_with_name[i]
            # remove speaker intro
            speech = re.sub(intro_regex, '', speech, flags=re.IGNORECASE).strip()
            # replace linebreaks with spaces
            speech = speech.replace("\n", " ")
            # replace multiple spaces with single space
            speeches_with_name[i] = re.sub('\\s+', ' ', speech)

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
        
        clean_speech = clean_speech.replace("… …", " ") # \u2026
        
        df.at[row_idx, "text"] = clean_speech
        
        print(f"Entry #{row_idx + 1} - Done successfully")

    df = df[~df.isnull().any(axis=1)]
    df.to_csv("data/out_with_text_processed.csv", index=False)

# %%
