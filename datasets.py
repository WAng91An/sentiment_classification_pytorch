from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np

def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item = item.split(",")
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict

def load_data(data_path,data_stop_path):
    data_list = open(data_path).readlines()[1:]
    stops_word = open(data_stop_path).readlines()
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(" ")
    stops_word.append("\n")
    data = []
    max_len_seq = 0
    np.random.shuffle(data_list)
    for item in data_list[:1000]:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content, cut_all=False)
        seg_res = []
        for seg_item in seg_list:
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)
        if len(seg_res) > max_len_seq:
            max_len_seq = len(seg_res)
        data.append([label, seg_res])
    return data, max_len_seq


class text_ClS(Dataset):
    def __init__(self, voc_dict_path, data_path, data_stop_path, max_len_seq=None):
        self.data_path = data_path
        self.data_stop_path = data_stop_path
        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_seq_len = load_data(self.data_path, self.data_stop_path)
        if max_len_seq is not None:
            self.max_seq_len = max_len_seq
        np.random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])
        if len(input_idx) < self.max_seq_len:
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_seq_len - len(input_idx))]
        data = np.array(input_idx)
        return label, data

def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)

# if __name__ == "__main__":
#     data_path = "/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/weibo_senti_100k.csv"
#     data_stop_path = "/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/hit_stopword"
#     dict_path = "/Users/wangruiqian/Documents/Code/Project/sentiment_classification_pytorch/data/dict"
#     dataset = text_ClS(dict_path, data_path, data_stop_path)
#     train_dataloader = data_loader(dataset, config)
#     for i, batch in enumerate(train_dataloader):
#         print(batch[1].size())