import jieba
import math
import os
import random
import re
import json
from collections import defaultdict
from calc_tfidf import calculate_tfidf, tf_idf_topk

"""
基于tfidf实现简单文本摘要

思路：
选取每天文章中3句tfidf值最高的句子，组成摘要
"""
jieba.initialize()


def load_data(path):
    corpus = []
    with open(path, encoding='utf-8') as f:
        documents = json.loads(f.read())
        for document in documents:
            assert "\n" not in documents
            assert "\n" not in document["content"]
            corpus.append(document["title"] + "\n" + document["content"])
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus


def generate_text_summarization(document_tf_idf, content, top=3):
    sentences = re.split("？|！|。", content) # 将一篇文章的内容切分成句子
    # 过滤掉正文在五句以内的文章
    if len(sentences) <= 5:
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence_score = 0
        words = jieba.lcut(sentence) # 将句子进行分词处理
        for word in words:
            sentence_score += document_tf_idf.get(word, 0) # 计算句子的tfidf分数
            sentence_score /= len(words) + 1
            result.append([sentence_score, index])
    result = sorted(result, key=lambda x: x[0], reverse=True)
    # 权重最高的可能依次是第10，第6，第3句，将他们调整为出现顺序比较合理，即3,6,10
    important_sentence_index = sorted([x[1] for x in result[:top]])
    return "。".join([sentences[index] for index in important_sentence_index])
        
    
def get_summarization(tf_idf_dict, corpus):
    res = []
    for index, document_tf_idf in tf_idf_dict.items():
        title, content = corpus[index].split("\n")
        summarization = generate_text_summarization(document_tf_idf, content)  # 通过文章内容来生成摘要
        if summarization is None:
            continue
        corpus[index] += "\n" + summarization
        res.append({"标题": title, "正文": content, "摘要": summarization})
    return res



if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    res = get_summarization(tf_idf_dict, corpus)
    writer = open("summarization.json", "w", encoding="utf8")
    writer.write(json.dumps(res, ensure_ascii=False, indent=2))
    writer.close()






