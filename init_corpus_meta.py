import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


class ARValuesMissingException(Exception):
    pass


def process_vote_detail(detail):
    votes = {}
    detail = detail.split("<BR>")
    if len(detail) == 1:  # all in favour
        in_favour = re.findall("\<I>(.*?)\</I>", detail[0])
        votes["a_favor"] = list(map(str.strip, in_favour))
        votes["contra"] = []
        votes["abstencao"] = []
    if len(detail) == 2:
        in_favour = re.findall("\<I>(.*?)\</I>", detail[0])
        votes["a_favor"] = list(map(str.strip, in_favour))
        other = re.findall("\<I>(.*?)\</I>", detail[1])
        if "Contra:" in detail[1]:
            votes["contra"] = list(map(str.strip, other))
            votes["abstencao"] = []
        elif "Abstenção:" in detail[1]:
            votes["abstencao"] = list(map(str.strip, other))
            votes["contra"] = []
        else:
            votes["abstencao"] = []
            votes["contra"] = []
    if len(detail) == 3:
        in_favour, against, abstention = detail
        in_favour = re.findall("\<I>(.*?)\</I>", in_favour)
        against = re.findall("\<I>(.*?)\</I>", against)
        abstention = re.findall("\<I>(.*?)\</I>", abstention)
        votes["a_favor"] = list(map(str.strip, in_favour))
        votes["contra"] = list(map(str.strip, against))
        votes["abstencao"] = list(map(str.strip, abstention))

    assert all(
        key in votes for key in ["a_favor", "contra", "abstencao"]
    ), f"{votes}---{detail}"

    return votes


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


def process_voting(votacao):
    row = []

    votacao = votacao[0]["pt_gov_ar_objectos_VotacaoOut"]

    if not all(key in votacao for key in ["resultado", "detalhe"]):
        raise ARValuesMissingException("Some vote required attributes are missing.")

    vot_resultado = votacao["resultado"]
    row.append(vot_resultado)
    vot_dict = process_vote_detail(votacao["detalhe"])
    row.append(vot_dict["a_favor"])
    row.append(vot_dict["contra"])
    row.append(vot_dict["abstencao"])

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
    row.append(pub_legislatura)
    pub_serie = publicacao["pubTipo"].split()[1]
    row.append(pub_serie)
    row.append(publicacao["pag"]["string"])
    row.append(f"dar_serie_{pub_serie}_{pub_legislatura}_{pub_sessao}_{pub_num:03}.pdf")
    row.append(scrape_first_page(publicacao))

    return row


def process_authors(deputado):
    row = []

    row.append(deputado["idCadastro"])
    row.append(deputado["nome"])
    row.append(deputado["GP"])
    # deputado2string = f"{deputado['idCadastro']}-{deputado['nome']}-{deputado['GP']}"
    # row.append(deputado2string)

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


leg_num_list = ["VIII", "IX", "X", "XI", "XII"]
# leg_num_list = ["XIII"]
rows = []
count = 0
for leg_num in leg_num_list:
    print(f"Processing Legislatura{leg_num}")
    with open(f"data/parliament/iniciativas/Iniciativas{leg_num}.json.txt") as f:
        json_data = json.load(f)

    iniciativas = json_data[
        "ArrayOfPt_gov_ar_objectos_iniciativas_DetalhePesquisaIniciativasOut"
    ]["pt_gov_ar_objectos_iniciativas_DetalhePesquisaIniciativasOut"]

    for i, iniciativa in enumerate(iniciativas):
        try:
            # skip iniciativas conjuntas
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
            count += 1

            for orador in speakers:
                row = []
                publicacao = list(get_value_from_key("publicacao", orador))
                publicacao = publicacao[0]["pt_gov_ar_objectos_PublicacoesOut"]

                row.extend(add_ini_attributes(iniciativa))
                row.extend(process_voting(votacao))
                row.extend(process_publication(publicacao))

                deputado = list(get_value_from_key("deputados", orador))
                if not deputado:
                    continue
                deputado = deputado[0]
                row.extend(process_authors(deputado))

                print("Adding row...")
                rows.append(row)

            print(f"Initiative #{i + 1} done")

        except ARValuesMissingException as e:
            print(f"Skipping initiative #{i + 1}. {e}")

print(f"Count {count}")


df = pd.DataFrame(
    rows,
    columns=[
        "ini_num",
        "ini_leg",
        "ini_tipo",
        "ini_titulo",
        "ini_sessao",
        "data_inicio_leg",
        "data_fim_leg",
        "vot_resultado",
        "vot_favor",
        "vot_contra",
        "vot_abstencao",
        "pub_num",
        "pub_sessao",
        "pub_data",
        "pub_legislatura",
        "pub_serie",
        "pages",
        "pdf_file_path",
        "doc_first_page",
        "dep_id",
        "dep_nome",
        "dep_gp",
    ],
)

df.to_csv("data/out.csv", index=False)
