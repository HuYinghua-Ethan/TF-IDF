import jieba
import math
import os
import json
from collections import defaultdict

"""
代码分解
text_tfidf_dict.items():
这个方法用于获取字典 text_tfidf_dict 中的所有键值对，以一种可遍历的形式返回。例如，如果字典内容是 {'word1': 0.5, 'word2': 0.3}，则 items() 将返回一个包含元组的视图：dict_items([('word1', 0.5), ('word2', 0.3)])。
sorted(..., key=lambda x: x[1], reverse=True):

sorted() 函数用于对可迭代对象进行排序。在这里，传入的可迭代对象是字典的项。
key=lambda x: x[1] 指定了排序的依据。x 代表字典中的每一个项（元组），x[1] 是元组的第二个元素，即 TF-IDF 值。所以排序将基于这些 TF-IDF 值进行。
reverse=True 表明我们希望以降序进行排序，也就是说，较高的 TF-IDF 值将排在前面。

结果:
排序完成后，word_list 将是一个列表，列表中的每个元素都是一个元组，格式为 (词, TF-IDF值)，并且按 TF-IDF 值从高到低排列。

text_tfidf_dict = {
    'apple': 0.6,
    'banana': 0.1,
    'cherry': 0.8,
    'date': 0.4
}

word_list = sorted(text_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
print(word_list)  # 输出: [('cherry', 0.8), ('apple', 0.6), ('date', 0.4), ('banana', 0.1)]
"""


# 统计tf和idf值
def build_tf_idf_dict(corpus):
    tf_dict = defaultdict(dict) # key:文档序号 value:dict，文档中每个词出现的频率
    idf_dict = defaultdict(set) # key:词   value:set, 文档序号，用于计算每个词在多少篇文档中出现过
    for text_index, text_words in enumerate(corpus):
        for word in text_words:
            if word not in tf_dict[text_index]:
                tf_dict[text_index][word] = 0
            tf_dict[text_index][word] += 1
            idf_dict[word].add(text_index) 
    idf_dict = dict([(key, len(value)) for key, value in idf_dict.items()])
    return tf_dict, idf_dict


# 计算tfidf值
def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = defaultdict(dict)
    for text_index, word_tf_count_dict in tf_dict.items():
        for word, tf_count in word_tf_count_dict.items():
            tf = tf_count / sum(word_tf_count_dict.values())
            # tfidf = 词频 × 逆文档频率
            tf_idf_dict[text_index][word] = tf * math.log(len(tf_dict)/(idf_dict[word]+1))
    return tf_idf_dict


#输入语料 list of string
#["xxxxxxxxx", "xxxxxxxxxxxxxxxx", "xxxxxxxx"]
def calculate_tfidf(corpus):
    # 先进行分词
    corpus = [jieba.lcut(text) for text in corpus]
    # print(corpus)
    tf_dict, idf_dict = build_tf_idf_dict(corpus)
    tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
    return tf_idf_dict
    

#根据tfidf字典，显示每个领域topK的关键词
def tf_idf_topk(tf_idf_dict, paths, top=10, print_word=True):
    topk_dict = {}
    for text_index, text_tfidf_dict in tf_idf_dict.items():
        word_list = sorted(text_tfidf_dict.items(), key=lambda x:x[1], reverse=True)
        topk_dict[text_index] = word_list[:top]
        if print_word:
            for i in range(top):
                print(word_list[i])
            print("--------------------------")
    return topk_dict


def main():
    dir_path = r"category_corpus/"  # 这是一个目录，里面包含了各种文档
    corpus = []
    paths = []
    for path in os.listdir(dir_path): # 列出目录中的所有文件和文件夹
        path = os.path.join(dir_path, path) # 组合出完整路径
        if path.endswith('txt'): # 筛选出txt文件
            corpus.append(open(path, encoding="utf-8").read())
            paths.append(os.path.basename(path))  # 返回路径中最后一部分的文件名。如果路径以 / 或 \ 结尾，返回空字符串。
    tf_idf_dict = calculate_tfidf(corpus)
    tf_idf_topk(tf_idf_dict, paths)


if __name__ == "__main__":
    main()


