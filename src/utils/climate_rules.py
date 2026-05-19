import unicodedata

MAPA_UF = {
    "AMAZONAS": "AM",
    "PARA": "PA",
    "ACRE": "AC",
    "MATO GROSSO": "MT",
    "GOIAS": "GO",
    "MINAS GERAIS": "MG"
}

def normalize_text(text):

    text = text.upper().strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ASCII",
        "ignore"
    ).decode(
        "ASCII"
    )

    return text

def name_to_uf(nome_uf):

    nome_normalizado = normalize_text(nome_uf)
    return MAPA_UF.get(nome_normalizado)

def estacao_seca(uf, mes):

    seco_centro_oeste = [5, 6, 7, 8, 9, 10]

    if uf in [
        "GO",
        "MT",
        "MG"
    ]:
        return int(mes in seco_centro_oeste)

    seco_norte = [6, 7, 8, 9]

    if uf in [
        "AM",
        "PA",
        "AC"
    ]:
        return int(mes in seco_norte)

    return 0
