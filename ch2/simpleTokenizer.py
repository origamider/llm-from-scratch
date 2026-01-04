import re

# 基本的なトークナイザを実装。エンコード、デコードメソッドを作成。
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # 辞書内包表記。int->strね。

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) #前処理。単語を分割。
        preprocessed = [it.strip() for it in preprocessed if it.strip()] # 余分な空白を剥がす
        ids = [self.str_to_int[str] for str in preprocessed]
        return ids

    def decode(self,ids):
        res = [self.int_to_str[id] for id in ids]
        res = " ".join(res) # 空白文字ベースで結合。
        res = re.sub(r'\s+([,.?!"()\'])', r'\1', res)
        return res
with open("the-verdict.txt","r",encoding='utf-8') as f:
    text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [it.strip() for it in preprocessed if it.strip()]
all_words = sorted(set(preprocessed))
vocab = {token:integer for integer,token in enumerate(all_words)} # index,valueを対応させる。enumerate=数え上げる
print(vocab)
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(text)
print(ids)


# tmp = tokenizer.decode(ids)
# print(tmp)