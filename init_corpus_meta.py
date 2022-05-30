# %%
import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

parties = ["PS", "PSD", "BE", "CDS-PP", "PEV", "PCP"]


class ARValuesMissingException(Exception):
    pass


def process_vote_detail(detail):
    votes = {"in_favour": [], "against": [], "abstention": []}
    detail = detail.split("<BR>")
    if len(detail) == 1:  # all in favour
        in_favour = re.findall("\<I>(.*?)\</I>", detail[0])
        votes["in_favour"] = list(map(str.strip, in_favour))
    if len(detail) == 2:
        in_favour = re.findall("\<I>(.*?)\</I>", detail[0])
        votes["in_favour"] = list(map(str.strip, in_favour))
        other = re.findall("\<I>(.*?)\</I>", detail[1])
        if "Contra:" in detail[1]:
            votes["against"] = list(map(str.strip, other))
        elif "Abstenção:" in detail[1]:
            votes["abstention"] = list(map(str.strip, other))
    if len(detail) == 3:
        in_favour, against, abstention = detail
        in_favour = re.findall("\<I>(.*?)\</I>", in_favour)
        against = re.findall("\<I>(.*?)\</I>", against)
        abstention = re.findall("\<I>(.*?)\</I>", abstention)
        votes["in_favour"] = list(map(str.strip, in_favour))
        votes["against"] = list(map(str.strip, against))
        votes["abstention"] = list(map(str.strip, abstention))

    def clean_noisy_party_labels(votes_dict, key):
        parties_regex = rf"^\d+-({'|'.join(parties)})$"
        votes_dict[key] = [i for i in votes[key] if not re.search(parties_regex, i)]

    clean_noisy_party_labels(votes, "in_favour")
    clean_noisy_party_labels(votes, "against")
    clean_noisy_party_labels(votes, "abstention")

    assert all(
        key in votes for key in ["in_favour", "against", "abstention"]
    ), f"{votes}---{detail}"

    return votes


def process_voting(votacao):
    row = []

    votacao = votacao[0]["pt_gov_ar_objectos_VotacaoOut"]

    if not all(key in votacao for key in ["resultado", "detalhe"]):
        raise ARValuesMissingException("Some vote required attributes are missing.")

    vot_resultado = votacao["resultado"]
    row.append(vot_resultado)
    vot_dict = process_vote_detail(votacao["detalhe"])
    row.append(vot_dict["in_favour"])
    row.append(vot_dict["against"])
    row.append(vot_dict["abstention"])

    return row


def scrape_first_page(publicacao):
    URL = publicacao["URLDiario"]
    r = requests.get(URL)

    soup = BeautifulSoup(r.content, "html.parser")

    pagination_html = soup.find("ul", {"class": "pagination"})

    if pagination_html is None:
        raise ARValuesMissingException(
            "The page does not seem to have a pagination footer."
        )

    children = pagination_html.findChildren("li", recursive=False)
    first_page = children[1].string

    return first_page


def add_ini_attributes(iniciativa):
    row = []
    ini_num = iniciativa["iniNr"]
    row.append(ini_num)
    ini_leg = iniciativa["iniLeg"]
    row.append(ini_leg)
    ini_tipo = iniciativa["iniDescTipo"]
    row.append(ini_tipo)
    ini_titulo = iniciativa["iniTitulo"]
    row.append(ini_titulo)
    ini_sessao = iniciativa["iniSel"]
    row.append(ini_sessao)
    dataInicioLeg = iniciativa["dataInicioleg"]
    row.append(dataInicioLeg)
    dataFimLeg = iniciativa["dataFimleg"]
    row.append(dataFimLeg)

    return row


def process_ini_authors(iniciativa):
    row = []

    if "iniAutorDeputados" in iniciativa:
        autores_deputados = iniciativa["iniAutorDeputados"][
            "pt_gov_ar_objectos_iniciativas_AutoresDeputadosOut"
        ]
        if isinstance(autores_deputados, list):
            autores_deputados = [
                f"{dep['idCadastro']}-{dep['nome']}-{dep['GP']}"
                for dep in autores_deputados
            ]
        else:
            autores_deputados = f"{autores_deputados['idCadastro']}-{autores_deputados['nome']}-{autores_deputados['GP']}"
    else:
        autores_deputados = iniciativa["iniAutorOutros"]["nome"]

    row.append(autores_deputados)

    return row


def process_publication(publicacao):
    row = []

    if not all(
        key in publicacao
        for key in ["pubNr", "pubSL", "pubdt", "pubLeg", "pubTipo", "pag"]
    ):
        raise ARValuesMissingException(
            "Some publication required attributes are missing"
        )

    pub_num = int(publicacao["pubNr"])
    row.append(pub_num)
    pub_sessao = int(publicacao["pubSL"])
    row.append(pub_sessao)
    row.append(publicacao["pubdt"])
    pub_legislatura = publicacao["pubLeg"]
    pub_serie = publicacao["pubTipo"].split()[1]
    pages = publicacao["pag"]["string"]
    if isinstance(pages, str):
        row.append([pages])
    else:
        row.append(pages)
    row.append(f"dar_serie_{pub_serie}_{pub_legislatura}_{pub_sessao}_{pub_num:03}.pdf")
    if pub_legislatura == "X":
        row.append(scrape_first_page(publicacao))
    else:
        row.append(1)

    return row


def process_speaker(deputado):
    row = []

    row.append(deputado["idCadastro"])
    row.append(deputado["nome"])
    row.append(deputado["GP"])

    return row


def get_value_from_key(key, dictionary):
    for k, v in (
        dictionary.items()
        if isinstance(dictionary, dict)
        else enumerate(dictionary)
        if isinstance(dictionary, list)
        else []
    ):
        if k == key:
            yield v
        elif isinstance(v, (dict, list)):
            for result in get_value_from_key(key, v):
                yield result


if __name__ == "__main__":
    leg_num_list = ["X", "XI", "XII"]  # "VIII", "IX"
    rows = []
    ini_count = 0
    for leg_num in leg_num_list:
        print(f"Processing Legislatura{leg_num}")
        with open(f"data/parliament/iniciativas/Iniciativas{leg_num}.json.txt") as f:
            json_data = json.load(f)

        iniciativas = json_data[
            "ArrayOfPt_gov_ar_objectos_iniciativas_DetalhePesquisaIniciativasOut"
        ]["pt_gov_ar_objectos_iniciativas_DetalhePesquisaIniciativasOut"]

        for i, iniciativa in enumerate(iniciativas):
            try:
                # skip joint initiatives
                conjuntas = get_value_from_key("iniciativasConjuntas", iniciativa)
                if list(conjuntas):
                    raise ARValuesMissingException("Joint initiatives.")

                discursos = list(
                    get_value_from_key(
                        "pt_gov_ar_objectos_peticoes_OradoresOut", iniciativa
                    )
                )
                if not discursos:
                    raise ARValuesMissingException("No speeches.")

                eventos = iniciativa["iniEventos"][
                    "pt_gov_ar_objectos_iniciativas_EventosOut"
                ]

                oradores, votacao = None, None
                print(f"{len(eventos)} events")
                for evento in eventos:
                    if evento["fase"] == "Discussão generalidade":
                        oradores = list(get_value_from_key("oradores", evento))

                    if evento["fase"] == "Votação na generalidade":
                        votacao = list(get_value_from_key("votacao", evento))

                if not oradores or not votacao:
                    print("Missing speaker or vote information.")
                    continue

                speakers = oradores[0]["pt_gov_ar_objectos_peticoes_OradoresOut"]
                if isinstance(speakers, dict):
                    speakers = [speakers]

                print(f"{len(speakers)} speakers")
                ini_count += 1

                for orador in speakers:
                    row = []
                    publicacao = list(get_value_from_key("publicacao", orador))
                    publicacao = publicacao[0]["pt_gov_ar_objectos_PublicacoesOut"]

                    row.extend(add_ini_attributes(iniciativa))
                    row.extend(process_ini_authors(iniciativa))
                    row.extend(process_voting(votacao))
                    row.extend(process_publication(publicacao))

                    deputado = list(get_value_from_key("deputados", orador))
                    if not deputado:
                        continue
                    deputado = deputado[0]
                    row.extend(process_speaker(deputado))

                    print("Adding row...")
                    rows.append(row)

                print(f"Initiative #{i + 1} done")

            except ARValuesMissingException as e:
                print(f"Skipping initiative #{i + 1}. {e}")

    print(f"# Initiatives: {ini_count}")


    df = pd.DataFrame(
        rows,
        columns=[
            "ini_num",
            "ini_leg",
            "ini_type",
            "ini_title",
            "ini_session",
            "leg_begin_date",
            "leg_end_date",
            "authors",
            "vot_results",
            "vot_in_favour",
            "vot_against",
            "vot_abstention",
            "pub_num",
            "pub_session",
            "pub_date",
            "pages",
            "pdf_file_path",
            "doc_first_page",
            "dep_id",
            "dep_name",
            "dep_parl_group",
        ],
    )

    df.to_pickle("data/initial_corpus_meta.pkl")  # df.to_csv("data/out.csv", index=False)

# %%
