"""Fixed matching patterns adapted from EDS-Pseudo v0.4.0 (BSD-3-Clause)."""

# See THIRD_PARTY_NOTICES.md for attribution and the original license.
# ruff: noqa: E501

PHONE = (
    r"("
    r"(?<!\d[ .-]{,3})(?<!\w)"
    r"((?:(?:\+|00)33\s?[.]\s?(?:\(0\)\s?[.]\s?)?|0)[1-9](?:(?:[.]\d{2}){4}|\d{2}(?:[.]\d{3}){2})(?![\d])"
    r"|(?:(?:\+|00)33\s?[-]\s?(?:\(0\)\s?[-]\s?)?|0)[1-9](?:(?:[-]\d{2}){4}|\d{2}(?:[-]\d{3}){2})(?![\d])"
    r"|(?:(?:\+|00)33\s?(?:[-]\s?)?(?:\(0\)\s?)?|0)[1-9](?:(?:[ ]?\d{2}){4}|\d{2}(?:[ ]?\d{3}){2})(?![\d]))"
    r"\b(?![ .-]{,3}\d)"
    r")"
)

MAIL = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?: ?\. ?[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*") ?@ ?(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])? ?\. ?)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) ?\. ?){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

SECU = r"""(?x)
# No digits just before on the same line
(?<!\d[ .-/+=_]{,3})\b
(
# Sex
    (?:[1-2])[ ]?
# Year of birth
    (?:([0-9][ ]?){2})[ ]?
# Month of birth
    (?:0[ ]?[0-9]|[2-35-9][ ]?[0-9]|[14][ ]?[0-2])[ ]?
# Location of birth
    (?:
        (?:
            0[ ]?[1-9]
            |[1-8][ ]?[0-9]
            |9[ ]?[0-69]
            |2[ ]?[abAB]
        )[ ]?
        (?:
            0[ ]?0[ ]?[1-9]|0[ ]?[1-9][ ]?[0-9]|
            [1-8][ ]?([0-9][ ]?){2}|9[ ]?[0-8][ ]?[0-9]|9[ ]?9[ ]?0
        )
        |(?:9[ ]?[78][ ]?[0-9])[ ]?(?:0[ ]?[1-9]|[1-8][ ]?[0-9]|9 ?0)
    )[ ]?
# Birth number 001-999
    (?:0[ ]?0[ ]?[1-9]|0[ ]?[1-9][ ]?[0-9]|[1-9][ ]?([0-9][ ]?){2})[ ]?
# Control key
    (?:0[ ]?[1-9]|[1-8][ ]?[0-9]|9[ ]?[0-7])
|
# Temporary NSS
    [3478][ ]?(?:[0-9][ ]?){14}
)
# Not followed by digits on the same line
\b(?![ .-]{,3}\d)
"""

SIMPLE_PATTERNS = {"TEL": PHONE, "MAIL": MAIL, "SECU": SECU}

_OPTIONAL_NEWLINE = r"[ ]*\n?[ ]*"
_STREET_BASE = [
    "avenue",
    "villa",
    "impasse",
    "ave",
    "rues?",
    "(?<!(?i:en|au|le)[ ])cours?",
    "(?<!(?i:en|de|la)[ ])place?",
    "parvis?",
    "(?<!(?i:en)[ ])routes?",
    "alll?ee?",
    "sentier",
    "faubourg",
    r"(?<!(?i:de)[ ])passage(?![ ](?i:en|a|au|aux)\b)",
    "boulevards?",
    "bd",
    "chemin",
    "quaie?s?",
    "residences?",
    "citee?",
    "square",
]
_STREET = [
    p.title().replace("(?I:", "(?i:").replace(r"\B", r"\b") for p in _STREET_BASE
] + _STREET_BASE
_STREET_UPPER = [p.upper().replace("(?I:", "(?i:").replace(r"\B", r"\b") for p in _STREET_BASE]
_STREET_PIECE = r"(?:(?:(?P<STREET_PIECE>[A-Z][A-Za-zéèà]{1,})|(?P<LOWER_STREET_PIECE>[a-zéèà]{2,})|de|du|la|a|à|le|les|des|DE|DU|LA|LE|LES|DES|D|L|d|l)\b|d'|D')"
_STREET_PIECE_UPPER = r"(?:(?:(?P<STREET_PIECE>[A-Z][A-Z]+)|[A-Z]\.|DE|DU|LA|LE|LES|DES|D|L)\b|D')"
_CITY = r"(?P<VILLE>(?:(?:[A-Z][A-Z]+|sur|en|Paris)[-]?)+(?<![-])(?:[ ]*(?i:CEDEX)[ ]*\d{{2}})?)\b"
_CITY_SPACED = (
    r"(?P<VILLE>(?:(?:[A-Z][A-Z]+|sur|en|Paris)[- ]?)+(?<![- ])(?:[ ]*(?i:CEDEX)[ ]*\d{{2}})?)\b"
)

ADDRESS_PATTERNS = [
    rf"""(?x)
(?<=(?P<TRIGGER>Adresse[ ]?:)?\s*)
(
        (?i:(?<![:])\b(?:(?P<NUMERO>[0-9]\d*)[,]?(?:\s*(?:bis|a|b|ter))?[ ]+)?)
        (?:
            \b(?P<UPPER_STREET>(?:{"|".join(_STREET_UPPER)})\b[ ]+)
            \b(?:{_STREET_PIECE_UPPER}[ -]*)*{_STREET_PIECE_UPPER}
        |
            \b(?:{"|".join(_STREET)})\b[ -]+
            \b(?:{_STREET_PIECE}[ -]*)*{_STREET_PIECE}
        )
    |
    (?<=(?P<TRIGGER>Ville[ ]?:)){_OPTIONAL_NEWLINE}
)
(?=
    (?i:(?:[ ]*[à,.-])?{_OPTIONAL_NEWLINE}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)\)?)
    {_OPTIONAL_NEWLINE}(?:{_CITY})
    |
    (?i:(?:[ ]*[à,.-])?{_OPTIONAL_NEWLINE}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)?\)?)
    (?:{_CITY})
    |
    (?:(?:[ ]*[à,.-])?[ ]*{_CITY})?
    (?i:{_OPTIONAL_NEWLINE}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})\)?|(?:[1-9]|1[0-9]|20)[èe]m?e?)?)
)
""",
    rf"""(?x)
(
        (?i:(?<![:])\b(?:(?P<NUMERO>[0-9]\d*)[,]?(?:\s*(?:bis|a|b|ter))?[ ]+)?)
        (?:
            \b(?:(?:{"|".join(_STREET + _STREET_UPPER)})\b[ -]+)
            \b(?:(?:[1-9A-Za-z]+)[ -,]*\n?)*(?:[1-9A-Za-z]+)
        )
        (?P<REGEX_2>)
)
(?=
    (?i:(?:[ ]*[à,.-]|[ ]*\n?)?[ ]*\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)\)?[ ]+)
    (?:{_CITY_SPACED})
    |
    (?:(?:[ ]*[à,.-]|[ ]*\n?)?\s*{_CITY_SPACED})
    (?i:[ ]+\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})\)?|(?:[1-9]|1[0-9]|20)[èe]m?e?))
)
""",
]
