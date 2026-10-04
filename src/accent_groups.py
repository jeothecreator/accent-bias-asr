"""
Accent grouping rules: how messy, self-reported Common Voice accent labels
become a small set of accent groups.

This file IS a research decision. Every choice here changes the results, so
every rule has a reason next to it, and prepare_dataset.py copies all of it
into data/DATA_CARD.md. Edit freely, then re-run prepare_dataset.py.

How a clip gets a group
-----------------------
1. The `accents` field is split on "|" (speakers can list several accents).
2. Each piece is normalized (lowercase, periods removed, spaces collapsed).
3. Each piece is checked against RULES top to bottom; the FIRST match wins.
   The result is a group name, EXCLUDE (deliberately left out), IGNORE
   (uninformative, e.g. "neutral"), L2_MARKER ("non native speaker"), or
   nothing (unmapped).
4. A clip is kept only if every informative piece lands in the SAME group.
   Any EXCLUDE, unmapped, or conflicting piece removes the clip. This is strict
   on purpose: a clip in the "us" group should come from someone who described
   their accent only as American.
5. An L2_MARKER piece supports the l2_european group but can't place a clip
   by itself: "Non native speaker|German English" -> l2_european,
   "Non native speaker" alone -> unmapped, "Non native speaker|United States
   English" -> conflict.

Framework: the groups loosely follow Kachru's "three circles" of English
(inner circle: US, England, Canada, Australia/NZ; outer circle: South Asia,
Southern Africa, Southeast Asia; expanding circle: second-language speakers).
"""

import re

EXCLUDE = "EXCLUDE"
IGNORE = "IGNORE"
L2_MARKER = "L2_MARKER"  # "non native", "foreign": only valid alongside a European first language

# Group id -> plain-language description used in the data card and charts.
GROUPS = {
    "us": "United States",
    "england": "England",
    "canada": "Canada",
    "australia_nz": "Australia and New Zealand",
    "south_asia": "India and South Asia",
    "southern_africa": "Southern Africa",
    "southeast_asia": "Philippines, Malaysia and Singapore",
    "l2_european": "Second-language English, European first language",
}

# (regex, result, reason). Checked in order; first match wins.
# Order matters: specific traps ("new england", "african american") must come
# before the generic patterns they would otherwise trigger ("england", "american").
RULES = [
    # ---- Uninformative pieces: dropped, they don't affect the clip ----
    (r"^(neutral|standard|normal|general|none|no accent|native|native speaker|"
     r"slight|slight accent|light|light accent|clear)( accent)?$",
     IGNORE, "Describes accent strength, not which accent"),

    (r"^(non[- ]?native|foreign|second language|esl|learner)( speaker| english| accent)?$",
     L2_MARKER, "Says the speaker is non-native but not which first language"),

    # ---- Deliberate exclusions, checked before any group ----
    (r"\bimpediment|\bstutter|\bstammer|\blisp\b|\brhotacism|\bdysarthria|\bhearing (loss|impair)|\bdeaf\b",
     EXCLUDE, "Speech or hearing difference, not an accent; would confound the accent comparison"),
    (r"\bsouth atlantic|\bfalkland|\bsaint helena|\bst helena|\bsouthatlandtic",
     EXCLUDE, "South Atlantic: too few speakers for its own group"),
    (r"\bmix|\bblend|\bhybrid|\bbetween\b|\bcombination\b|\bmulti",
     EXCLUDE, "Self-described mixed accent: can't be placed in one group"),
    (r"\bafrican[- ]american\b|\baave\b|\bblack american\b",
     EXCLUDE, "A distinct US variety; too few clips to stand alone and should not be silently merged into 'us'"),
    (r"\b(native american|american indian|first nations)\b",
     EXCLUDE, "Contains 'american'/'indian' but is not General American or South Asian"),
    (r"\bwest indi|\bcaribbean|\bjamaica|\btrinidad|\bbahamas|\bbermuda|\bbarbad|\bguyan",
     EXCLUDE, "Caribbean English: not one of the studied groups (and 'west indian' would wrongly match 'indian')"),
    (r"\b(south|latin|central) america|\bmexic|\bbrazil|\bargentin|\bcolombia|\bchile|\bperu\b|\bvenezuel",
     EXCLUDE, "Latin American: contains 'american' but is not US English; L1 not consistently stated"),
    (r"\bbritish columbia\b",
     "canada", "Canadian province; must come before the 'british' exclusion"),
    (r"\bnorth american\b",
     EXCLUDE, "Ambiguous between United States and Canada"),
    (r"\bbritish\b|\b(uk|united kingdom)\b",
     EXCLUDE, "Ambiguous: could be England, Scotland, Wales or Northern Ireland"),
    (r"\bscot|\bwelsh\b|\bwales\b|\birish\b|\bireland\b",
     EXCLUDE, "Scottish/Welsh/Irish: out of scope here (could become groups if speaker counts allow)"),
    (r"\bhong kong|\bhongkong|\bchin(a|ese)\b|\bjapan|\bkorea|\btaiwan|\bvietnam|\bthai",
     EXCLUDE, "East Asian: out of scope for this group set"),
    (r"\bnigeria|\bghana|\bkenya|\buganda|\btanzania|\bwest africa|\beast africa|\bethiopia|^african$",
     EXCLUDE, "Other African varieties: not pooled with Southern Africa to avoid treating Africa as one accent"),

    # ---- French Canadian before 'canada': these speakers are mostly L2 English ----
    (r"\bfrench[- ]canad|\bquebec|\bquébec|\bquebecois|\bquébécois",
     "l2_european", "Quebec French first language: second-language English, not Canadian English"),

    # ---- Inner circle ----
    (r"\bnew england\b",
     "us", "US region; must come before the 'england' rule"),
    (r"\bunited states\b|\b(us|usa)\b|\bamerican?\b|\bgeneral american\b|\bmidwest|\bnew york|"
     r"\bcalifornia|\btexas|\bboston|\bchicago|\bpacific northwest|\bappalachia|\bsouthern (us|american)\b|"
     r"\bminnesota|\bwisconsin|\bnew orleans|\blouisiana|\bohio\b|\bpennsylvania|\bphiladelphia|"
     r"\bnew jersey|\bjersey\b|\bflorida|\bcarolina|\bvirginia|\btennessee|\bkentucky|\balabama|"
     r"\bmississippi|\boklahoma|\bkansas|\bmissouri|\biowa\b|\bmichigan|\bseattle|\bcolorado|\butah\b|"
     r"\barizona|\bhawai|\balaska|\bbaltimore|\bmaryland|\bmassachusetts|\bconnecticut|\bbrooklyn|"
     r"\bbronx|\bdeep south\b|\bwest coast\b|\beast coast\b",
     "us", "United States, including US regional descriptions"),
    (r"\bengland\b|\blondon|\bcockney|\byorkshire|\bmanchester|\bmancunian|\bscouse|\bliverpool|"
     r"\bgeordie|\bbirmingham|\bbrummie|\bmidlands|\bwest country|\bessex|\breceived pronunciation\b|"
     r"\brp\b|\b(northern|southern) english\b|\bestuary|\blancashire|\bhome counties|\bcornwall|"
     r"\bcornish|\bdevon|\bkent\b|\bsomerset|\bnorfolk|\bsuffolk|\bcumbria|\bnorthumb|\bbristol|"
     r"\bleeds\b|\bsheffield|\bnottingham|\bwest midlands|\beast anglia|\bsussex|\bsurrey|\bhampshire|"
     r"\bdorset|\bwiltshire|\boxford|\bcambridge|\bblack country|\bteesside|\bwearside",
     "england", "England, including English regional descriptions"),
    (r"\bcanad|\bontario|\btoronto|\bvancouver|\balberta|\bmanitoba|\bsaskatchewan|\bnova scotia|"
     r"\bnewfoundland|\bottawa|\bcalgary|\bwinnipeg",
     "canada", "Canada, including provinces and cities"),
    (r"\baustralia|\baussie|\bnew zealand|\bnewzealand|\bkiwi\b|\bnz\b",
     "australia_nz", "Australia and New Zealand: closely related varieties, pooled for enough speakers"),

    # ---- Outer circle ----
    (r"\bindia|\bsouth asia|\bpakistan|\bsri lanka|\bbangladesh|\bnepal|\bhindi|\btamil|\btelugu|"
     r"\bbengali|\bpunjabi|\bmarathi|\bgujarati|\bkannada|\bmalayalam|\burdu",
     "south_asia", "India and South Asia (matches Common Voice's own predefined label)"),
    (r"\bsouth(ern)? africa|\bzimbabw|\bnamibia|\bbotswana|\bzambia|\bafrikaans|\bzulu|\bxhosa",
     "southern_africa", "Southern Africa (matches Common Voice's own predefined label)"),
    (r"\bfilipin|\bphilippin|\bpinoy|\btagalog|\bmalaysia|\bsingapore|\bsinglish",
     "southeast_asia", "Philippines, Malaysia, Singapore: English widely used in education and government"),

    # ---- Expanding circle: defined by first language, not country ----
    (r"\bgerman|\bfrench|\bspanish|\bitalian|\bdutch|\bportuguese|\bpolish|\bpoland|\brussia|"
     r"\bukrain|\bswed|\bnorw|\bdanish|\bdenmark|\bfinnish|\bfinland|\bczech|\bslovak|\bhungar|"
     r"\broman(ia|ian)\b|\bbulgaria|\bgreek|\bgreece|\bserbia|\bcroatia|\bbosnia|\bsloven|\blithuania|"
     r"\blatvia|\bestonia|\bbelarus|\bcatalan|\bbasque|\bswiss|\baustria|\bbelgi|\bflemish|\beuropean|"
     r"\beurope\b|\bslavic|\bscandinav|\bnordic|\bicelandic|\balemannic|\bbavarian",
     "l2_european", "Second-language English speakers whose first language is European"),
]

_COMPILED = [(re.compile(pattern), result, reason) for pattern, result, reason in RULES]


def normalize_piece(piece: str) -> str:
    piece = piece.lower().replace(".", "")
    return re.sub(r"\s+", " ", piece).strip(" ,;:-")


def classify_piece(piece: str):
    """Return (result, rule_index) for one accent piece; (None, None) if unmapped."""
    for i, (pattern, result, _reason) in enumerate(_COMPILED):
        if pattern.search(piece):
            return result, i
    return None, None


def classify_label(raw: str):
    """
    Classify a full raw `accents` value.
    Returns (status, group):
      status is one of: "grouped", "blank", "uninformative", "excluded",
      "unmapped", "conflict"; group is set only when status == "grouped".
    """
    pieces = [normalize_piece(p) for p in str(raw).split("|")]
    pieces = [p for p in pieces if p]
    if not pieces:
        return "blank", None

    groups = set()
    unmapped = excluded = l2_marker = False
    for piece in pieces:
        result, _ = classify_piece(piece)
        if result is None:
            unmapped = True
        elif result == EXCLUDE:
            excluded = True
        elif result == L2_MARKER:
            l2_marker = True
        elif result != IGNORE:
            groups.add(result)

    if excluded:
        return "excluded", None
    if unmapped:
        return "unmapped", None
    if len(groups) > 1 or (l2_marker and groups - {"l2_european"}):
        return "conflict", None
    if not groups:
        return ("unmapped" if l2_marker else "uninformative"), None
    return "grouped", groups.pop()
