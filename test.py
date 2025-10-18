from pythainlp import word_tokenize
text = "สมชายไปมหาวิทยาลัยสงขลานครินทร์"
print(word_tokenize(text))

from pythainlp import word_tokenize

text = "สมชายไป Prince of Songkla University"
print(word_tokenize(text, engine="newmm", keep_whitespace=False))
