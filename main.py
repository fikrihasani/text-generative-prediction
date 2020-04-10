import pandas as pd
import model
import os
df = pd.read_csv('data/alls.csv')
df["Content"] = df["Content"].str.lower()
df["Content"] = df["Content"].str.replace('[^\w\s]', '')
text = " ".join(" ".join(df["Content"].tolist()).split())
print(df.head())
print(text[:200])

model.preprocessing(text)
if not os.path.exists("model"):
    os.mkdir("model")
    model.feature_build()
    model.build()

lstm_model = model.load()
chars = sorted(list(set(text)))
text = 'the present study is a history of the dewey'
for subtext in range(0, len(text)-3, 1):
    now_subtext = text[subtext:subtext+3]
    predict = model.predict_completions(lstm_model, now_subtext, chars)
    print(f"text: {now_subtext}: {predict}")
# print(predict)
