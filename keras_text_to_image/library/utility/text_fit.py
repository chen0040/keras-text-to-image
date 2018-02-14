from collections import Counter
import nltk


def fit_text(data, max_vocab_size, max_allowed_seq_length=None):
    counter = Counter()
    max_seq_length = 0
    for t in data:
        _, txt = t
        txt = 'START ' + txt.lower() + ' END'
        words = nltk.word_tokenize(txt)
        words = [word for word in words if word.isalnum()]
        seq_length = len(words)
        if max_allowed_seq_length is not None and seq_length > max_allowed_seq_length:
            seq_length = max_allowed_seq_length
            words = words[:seq_length-1] + ['END']
        max_seq_length = max(seq_length, max_seq_length)

        for w in words:
            counter[w] += 1

    word2idx = dict()
    for idx, word in enumerate(counter.most_common(max_vocab_size)):
        word2idx[word[0]] = idx

    config = dict()
    config['max_seq_length'] = max_seq_length
    config['word2idx'] = word2idx
    config['vocab_size'] = len(word2idx)
    config['idx2word'] = dict([(idx, word) for word, idx in word2idx.items()])

    return config
