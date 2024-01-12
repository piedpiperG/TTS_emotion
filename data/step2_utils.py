import re
from fron.frontend_cn import split_py, tn_chinese
from fron.frontend_en import read_lexicon, G2p
from fron.frontend import contains_chinese, re_digits, g2p_cn


def onetime(resource, sample):
    text = sample["text"]
    # del sample["original_text"]

    phoneme = get_phoneme(text, resource["g2p"]).split()

    sample["text"] = phoneme
    # sample["original_text"]=text
    sample["prompt"] = sample["original_text"]

    return sample


def onetime2(resource, sample):
    text = sample["original_text"]
    del sample["original_text"]
    try:
        phoneme = g2p_cn_en(text, resource["g2p_en"], resource[
            "lexicon"]).split()  # g2p_cn_eng_mix(text, resource["g2p_en"], resource["lexicon"]).split()
    except:
        print("Warning!!! phoneme get error! " + \
              "Please check text")
        print("Text is: ", text)
        return ""

    if not phoneme:
        return ""

    sample["text"] = phoneme
    sample["original_text"] = text
    sample["prompt"] = sample["original_text"]

    return sample


def get_phoneme(text, g2p):
    special_tokens = {"#0": "sp0", "#1": "sp1", "#2": "sp2", "#3": "sp3", "#4": "sp4", "<sos/eos>": "<sos/eos>"}
    phones = []

    for ph in text:
        if ph not in special_tokens:
            phs = g2p(ph)
            phones.extend([ph for ph in phs if ph])
        else:
            phones.append(special_tokens[ph])

    return " ".join(phones)


# re_english_word = re.compile('([a-z\-\.\']+|\d+[\d\.]*)', re.I)
re_english_word = re.compile(
    '([^\u4e00-\u9fa5]+|[ \u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09\u4e00-\u9fa5]+)',
    re.I)


def g2p_cn_en(text, g2p, lexicon):
    # Our policy dictates that if the text contains Chinese, digits are to be converted into Chinese.
    text = tn_chinese(text)
    parts = re_english_word.split(text)
    parts = list(filter(None, parts))
    tts_text = ["<sos/eos>"]
    chartype = ''
    text_contains_chinese = contains_chinese(text)
    for part in parts:
        if part == ' ' or part == '': continue
        if re_digits.match(part) and (text_contains_chinese or chartype == '') or contains_chinese(part):
            if chartype == 'en':
                tts_text.append('eng_cn_sp')
            phoneme = g2p_cn(part).split()[1:-1]
            chartype = 'cn'
        elif re_english_word.match(part):
            if chartype == 'cn':
                if "sp" in tts_text[-1]:
                    ""
                else:
                    tts_text.append('cn_eng_sp')
            phoneme = get_eng_phoneme(part, g2p, lexicon).split()
            if not phoneme:
                # tts_text.pop()
                continue
            else:
                chartype = 'en'
        else:
            continue
        tts_text.extend(phoneme)

    tts_text = " ".join(tts_text).split()
    if "sp" in tts_text[-1]:
        tts_text.pop()
    tts_text.append("<sos/eos>")

    return " ".join(tts_text)


def get_eng_phoneme(text, g2p, lexicon):
    """
    english g2p
    """
    filters = {",", " ", "'"}
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:

            for ph in lexicon[w.lower()]:
                if ph not in filters:
                    phones += ["[" + ph + "]"]

            if "sp" not in phones[-1]:
                phones += ["engsp1"]
        else:
            phone = g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():

                for ph in phone:
                    if ph not in filters:
                        phones += ["[" + ph + "]"]
                    if ph == " " and "sp" not in phones[-1]:
                        phones += ["engsp1"]
            elif phone == " ":
                continue
            elif phones:
                phones.pop()  # pop engsp1
                phones.append("engsp4")
    if phones and "engsp" in phones[-1]:
        phones.pop()

    return " ".join(phones)
