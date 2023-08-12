import argparse


def arg_parse_from_commandline(argNameList):
    parser = argparse.ArgumentParser()
    for argName in argNameList:
        parser.add_argument(argName, help=argName)
    args = parser.parse_args()
    return args


def tokenize_with_label(sentence, sscResult, tokenizer):
    tokenized_input = tokenizer(
        title_abs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # input['input_ids'][0](トークナイズされた入力文)と同じ長さで、その位置のトークンのラベルを格納するリスト
    label_list_for_words = [None for i in range(len(
        tokenized_input['input_ids'][0].tolist()))]

    # titleの位置にラベルを格納する
    # SEPトークンの位置を特定する
    sep_pos = input['input_ids'][0].tolist().index(102)
    for i in range(1, sep_pos):
        label_list_for_words[i] = 'title'

    # 各トークンの観点をlabel_positionsに格納
    for text_label_pair in ssc:
        text = text_label_pair[0]
        label = text_label_pair[1]

        # 1文単位でtokenizeする
        tokenizedText = tokenizer(
            text,
            return_tensors="pt",
            max_length=512
        )
        # 先頭の101([CLS])と末尾の102([SEP])を取り除く
        tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()

        start, end = find_subarray(
            input['input_ids'][0].tolist(), tokenizedText_input_ids)
        for i in range(start, end+1):
            label_list_for_words[i] = label

    return tokenized_input, label_list_for_words
