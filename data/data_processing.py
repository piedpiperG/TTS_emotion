import argparse
import os
import soundfile as sf
import librosa
import jsonlines
from tqdm import tqdm
import re


# 从原始数据中提取和处理文本和音频信息，并将这些信息以一种格式化的方式保存，便于后续的数据处理和机器学习任务
def step1():
    # 直接指定数据目录的路径
    ROOT_DIR = os.path.abspath(".")  # 替换为你的数据目录路径
    RAW_DIR = f"{ROOT_DIR}/raw"
    WAV_DIR = f"{ROOT_DIR}/wavs"
    TEXT_DIR = f"{ROOT_DIR}/text"

    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    with open(f"{RAW_DIR}/BZNSYP/ProsodyLabeling/000001-010000.txt", encoding="utf-8") as f, \
            jsonlines.open(f"{TEXT_DIR}/data.jsonl", "w") as fout1:

        lines = f.readlines()
        # 每两行作为一个处理单元，第一行包含文本，第二行包含音节信息。
        for i in tqdm(range(0, len(lines), 2)):
            key = lines[i][:6]

            # 使用正则表达式处理文本，去除特定标点符号。
            ### Text
            content_org = lines[i][7:].strip()
            content = re.sub("[。，、“”？：……！（ ）—；]", "", content_org)
            content_org = re.sub("#\d", "", content_org)

            chars = []
            prosody = {}
            j = 0
            while j < len(content):
                if content[j] == "#":
                    prosody[len(chars) - 1] = content[j: j + 2]
                    j += 2
                else:
                    chars.append(content[j])
                    j += 1

            # 处理特定的文本情况，如对特定的key值做特殊处理。
            if key == "005107":
                lines[i + 1] = lines[i + 1].replace(" ng1", " en1")
            if key == "002365":
                continue

            syllable = lines[i + 1].strip().split()
            s_index = 0
            phones = []
            phone = []

            # 解析音节信息，处理儿化音，构造发音单元列表。
            for k, char in enumerate(chars):
                # 儿化音处理
                er_flag = False
                if char == "儿" and (s_index == len(syllable) or syllable[s_index][0:2] != "er"):
                    er_flag = True
                else:
                    phones.append(syllable[s_index])
                    # phones.extend(lexicon[syllable[s_index]])
                    s_index += 1

                if k in prosody:
                    if er_flag:
                        phones[-1] = prosody[k]
                    else:
                        phones.append(prosody[k])
                else:
                    phones.append("#0")

            # 处理音频
            ### Wav
            path = f"{RAW_DIR}/BZNSYP/Wave/{key}.wav"
            wav_path = f"{WAV_DIR}/{key}.wav"
            y, sr = sf.read(path)
            # 使用 librosa 对音频进行重采样，将采样率转换为16000Hz。
            y_16 = librosa.resample(y, orig_sr=sr, target_sr=16_000)
            # 将处理后的音频数据写入新的文件中。
            sf.write(wav_path, y_16, 16_000)
            # 将处理后的文本和音频路径信息输出到JSONL格式的文件中
            fout1.write({
                "key": key,
                "wav_path": wav_path,
                "speaker": "BZNSYP",
                "text": ["<sos/eos>"] + phones[:-1] + ["<sos/eos>"],
                "original_text": content_org,
            })

    return


# 接下来还需要将文本转换为音素
def step2():
    args = argparse.Namespace(data_dir='.', generate_phoneme=False)
    ROOT_DIR = args.data_dir
    TRAIN_DIR = f"{ROOT_DIR}/train"
    VALID_DIR = f"{ROOT_DIR}/valid"
    TEXT_DIR = f"{ROOT_DIR}/text"

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)

    lexicon = read_lexicon(f"../lexicon/librispeech-lexicon.txt")

    g2p = G2p()

    resource = {
        "g2p": split_py,
        "g2p_en": g2p,
        "lexicon": lexicon,
    }

    with jsonlines.open(f"{TEXT_DIR}/data.jsonl") as f:
        data = list(f)

    new_data = []
    with jsonlines.open(f"{TEXT_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(data):
            if not args.generate_phoneme:
                sample = onetime(resource, sample)
            else:
                sample = onetime2(resource, sample)
            if not sample:
                continue
            f.write(sample)
            new_data.append(sample)

    with jsonlines.open(f"{TRAIN_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(new_data[:-3]):
            f.write(sample)

    with jsonlines.open(f"{VALID_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(data[-3:]):
            f.write(sample)

    return


if __name__ == "__main__":
    # 对文本和音频进行预处理
    step1()
