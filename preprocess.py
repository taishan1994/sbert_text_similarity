import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    data_file = '/data02/gob/data/text_similarity/蚂蚁金融/atec_nlp_sim_train_all.csv'
    data = pd.read_csv(data_file, sep='\t', header=None, names=['index', 's1', 's2', 'label'])
    # 获取数据和标签
    x = data[['s1', 's2']].values.tolist()
    y = data['label'].values.tolist()
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
    print(x_train[0], y_train[0])
    print('总共有数据：{}条，其中正样本：{}条，负样本：{}条'.format(
        len(x), sum(y), len(x) - sum(y)))
    print('训练数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_train), sum(y_train), len(x_train) - sum(y_train)))
    print('测试数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_test), sum(y_test), len(x_test) - sum(y_test)))
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()
