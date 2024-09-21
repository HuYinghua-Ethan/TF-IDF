import jieba
import math
import os
import json
from collections import defaultdict
from calc_tfidf import calculate_tfidf, tf_idf_topk
"""
基于tfidf实现简单搜索引擎

有用方法，特别是在处理动态或复杂的文本分词任务时。jieba.initialize() 是确保分词器正确设置和更新的一个有用方法，特别是在处理动态或复杂的文本分词任务时。
"""

jieba.initialize()



def load_data(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        documents = json.loads(f.read())
        for document in documents:
            corpus.append(document["title"] + "\n" + document["content"])
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus

def search_engine(query, tf_idf_dict, corpus, top=3):
    query_words = jieba.lcut(query)  #先对要查询的内容进行分词
    res = []
    for doc_id, td_idf in tf_idf_dict.items():
        score = 0  # 每篇文章都要更新一下 score
        for word in query_words:
            score += td_idf.get(word, 0)  # 计算每个词的tf-idf值,然后相加得到一个分数
        res.append([doc_id, score])
    res = sorted(res, key=lambda x:x[1], reverse=True)
    for i in range(top):
        doc_id = res[i][0]
        print(corpus[doc_id])
        print("-" * 10)
        

if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    while True:
        query = input("请输入您要搜索的内容: ")
        search_engine(query, tf_idf_dict, corpus)


