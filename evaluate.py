import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util
from preprocess import get_data

model_path = '/data02/gob/model_hub/sentence_hfl_chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)

x_train, x_test, y_train, y_test = get_data()
s1 = np.array(x_test)[:, 0]
s2 = np.array(x_test)[:, 1]
embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)
pre_labels = [0] * len(s1)
predict_file = open('predict.txt', 'w')
for i in range(len(s1)):
    similarity = util.cos_sim(embedding1[i], embedding2[i])
    if similarity > 0.5:
        pre_labels[i] = 1
    predict_file.write(s1[i] + ' ' +
                       s2[i] + ' ' +
                       str(y_test[i]) + ' ' +
                       str(pre_labels[i]) + '\n')
print(classification_report(y_test, pre_labels))
predict_file.close()
