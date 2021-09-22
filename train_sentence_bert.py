from torch.utils.data import  DataLoader
import torch.nn as nn
from sentence_transformers import  SentenceTransformer, InputExample, losses
from sentence_transformers import models, evaluation
from preprocess import get_data

model_path = '/data02/gob/model_hub/hfl_chinese-roberta-wwm-ext/'
word_embedding_model = models.Transformer(model_path, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=256, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

x_train, x_test, y_train, y_test = get_data()
train_examples = []
for s, label in zip(x_train, y_train):
    s1, s2 = s
    train_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
test_examples = []
for s, label in zip(x_test, y_test):
    s1, s2 = s
    test_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
train_loader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.CosineSimilarityLoss(model)

model_save_path = '/data02/gob/model_hub/sentence_hfl_chinese-roberta-wwm-ext/'
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
model.fit(train_objectives=[(train_loader, train_loss)],
          epochs=1,
          evaluator=evaluator,
          warmup_steps=100,
          save_best_model=True,
          output_path=model_save_path,)