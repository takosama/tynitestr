from tokenizers import Tokenizer, decoders

tok = Tokenizer.from_file("./tokenizer.json")
tok.decoder = decoders.ByteLevel()  # 念のため明示

text = "わたしはりんちゃです"

enc = tok.encode(text)
print(enc.ids)       # <- IDのリスト
print(enc.tokens)    # <- ByteLevel表現なので “変な文字” に見えるのは仕様

# ★ decode は enc.ids を渡す。Encodingオブジェクトを直接渡さない！
print(tok.decode(enc.ids, skip_special_tokens=True))


