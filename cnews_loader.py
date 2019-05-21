# -*- coding:utf-8 -*-
import sys
import numpy as np

import tensorflow.contrib.keras as kr
from collections import Counter
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def read_file(filename):
    """读取文件数据"""
    contents,labels=[],[]
    with open(filename,encoding="utf-8") as f:
        for line in f:
            try:
                label,content=line.strip().split("\t")
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents,labels

def build_vocab(train_dir,vocab_dir,vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train,_=read_file(train_dir)
    all_data=[]
    for content in data_train:
        all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    # [(5, 3), (7, 2), (8, 2), (4, 1), (6, 1), (76, 1)]
    # (5, 7, 8, 4, 6, 76)
    words = ['<PAD>'] + list(words)
    with open(vocab_dir,'w') as f:
        f.write('\n'.join(words)+'\n')
    # open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
def read_catergory():
    """d读取文件分类目录，固定"""
    categories=['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id
def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir,'r',encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))#一定要记住转为字典形式
    return words, word_to_id

def process_file(filename, word_to_id, cat_to_id,max_length=600):
    contents,labels=read_file(filename)
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[word] for word in contents[i] if word in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    y_pad=kr.utils.to_categorical(label_id,num_classes=len(cat_to_id))

    return x_pad,y_pad

def batch_iter(x,y,batch_size):
    """生成批次数据"""
    data_len=len(x)
    num_batch=int((data_len - 1) / batch_size) + 1
    indices=np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
if __name__ == '__main__':
    import numpy as np
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = np.array([[4], [5], [6], [7], [8],[5],[5],[76],[7],[8]])
    # counter = Counter(c)
    # count_pairs = counter.most_common(8)
    # words, _ = list(zip(*count_pairs))
    # words = ['<PAD>'] + list(words)
    # print(count_pairs)
    # print(words)
    import tensorflow.contrib.keras as kr
    import tensorflow as tf
    x_pad=kr.preprocessing.sequence.pad_sequences(c,20)
    print(x_pad)
    # sess=tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(x_pad))