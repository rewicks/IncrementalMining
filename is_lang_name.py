# pip3 install langcodes
# pip3 install language_data
# pip3 install unidecode

import langcodes
import language_data
import unidecode

# Given string txt, return tuple of booleans indicating
# whether the string matches either language name in either language
# So ("English", 'en', 'fr') would return (True, False), as would
#    ("Anglais", 'en', 'fr').
# Similarly, ("French", 'en', 'fr') would return (False, True), as would
#             ("Francais", 'en', 'fr')

# Remove diacritics, and lowercase
def NormalizeLangName(normalize_me):
    return unidecode.unidecode(normalize_me).lower()

      
def TextIsLangName(txt, lang1, lang2):
    l1 = langcodes.Language.get(lang1)
    l2 = langcodes.Language.get(lang2)
    norm_txt = NormalizeLangName(txt)
    lang1InLang1 = NormalizeLangName(l1.display_name(lang1))
    lang1InLang2 = NormalizeLangName(l1.display_name(lang2))
    lang2InLang1 = NormalizeLangName(l2.display_name(lang1))
    lang2InLang2 = NormalizeLangName(l2.display_name(lang2))
    IsLang1 = (norm_txt == lang1InLang1 or norm_txt == lang1InLang2 or norm_txt == lang1)
    IsLang2 = (norm_txt == lang2InLang1 or norm_txt == lang2InLang2 or norm_txt = lang1)
    return (IsLang1, IsLang2)
