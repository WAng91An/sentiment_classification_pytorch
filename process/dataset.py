import jieba

data_path = "/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/weibo_senti_100k.csv"
data_stop_path = "/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/hit_stopword"
data_list = open(data_path).readlines()[1:]
data_stop_list = open(data_stop_path).readlines()
data_stop_list = [i.strip() for i in data_stop_list]
data_stop_list.append(" ")
data_stop_list.append("\n")
print(data_stop_list)

voc_dict = {} # 根据语料库中的词构建字典，key：word ， value：频次
min_seq = 1 # 过滤掉字典中词出现小于等于 1 次的词
top_n = 1000 # 字典的最大长度
UNK = "<UNK>" # unknow
PAD = "<PAD>" # 问题建模时，扩展成同等长度的向量

for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.cut(content, cut_all=False)

    seg_word_list = []
    for seg_word in seg_list:
        if seg_word in data_stop_list:
            continue
        seg_word_list.append(seg_word)
        if seg_word in voc_dict.keys():
            voc_dict[seg_word] = voc_dict[seg_word] + 1
        else:
            voc_dict[seg_word] = 1

    print("分词前的句子：", content)
    print("分词后的句子：", seg_word_list)

print("老词典:", voc_dict)
# 过滤掉字典中出现频率小于等于 min_seq 的词，并且按照出现的频率从大到小排序，取前 top_n 词作为词典
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq], key=lambda x: x[1], reverse=True)[:top_n]
# 重新构建字典，根据词出现频率排序后的字典中词的位置设置新的索引
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}

print("构建的新词典：",voc_dict)

voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})

print("添加 UNK，PAD 元素", voc_dict)

# 写入词典信息
ff = open("/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/dict", "w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
ff.close()