# sbert_text_similarity
使用sentence-transformers（SBert）训练自己的文本相似度数据集并进行评估。预训练的bert可以去hugging face上面下载：chinese-roberta-wwm-ext。已经训练好的模型：
阿里云盘链接：https://www.aliyundrive.com/s/N2d11KphazT <br>
记得修改相关文件的路径。


# 说明
本文使用的数据是蚂蚁金融文本相似度数据集，具体数据在data文件夹下。<br>
--preprocess.py：获取数据，并划分训练集和测试集。<br>
--train_sentence_bert.py：使用自己的数据集利用sbert训练文本相似度任务。<br>
--evaluate.py：评估训练好的模型。<br>
--predict.txt：保存预测的结果。<br>

# 结果
```python
['开通花呗连接', '怎么开通了花呗'] 0
总共有数据：102477条，其中正样本：18685条，负样本：83792条
训练数据：71733条,其中正样本：13141条，负样本：58592条
测试数据：30744条,其中正样本：5544条，负样本：25200条
              precision    recall  f1-score   support

           0       0.88      0.93      0.90     25200
           1       0.57      0.43      0.49      5544

    accuracy                           0.84     30744
   macro avg       0.73      0.68      0.70     30744
weighted avg       0.83      0.84      0.83     30744
```
predict.txt：部分结果
```python
怎样修改支付宝花呗邮箱 支付宝的花呗怎么换号 0 0
花呗收款需要手续费吗 收信用卡蚂蚁花呗有没有手续费 0 0
为什么我借呗额度没恢复 我昨天还了蚂蚁借呗，为什么额度恢复不了 0 0
换了手机怎么没了借呗 支付宝账号换了新手机号 和花呗的手机号不同该怎么办 0 0
我的支付宝账号换了，怎么用花呗 我绑定的花呗是以前用的号码，现在换号码了。怎么绑定现在的号码 0 0
什么意思怎么老是扣款还说还款我也没怎么用这个花呗里面的钱 我没有贷款，怎么把我的钱都扣到花呗那里去什么意思 0 0
为什么我开通了花呗但是不能使用 为什么我的花呗借呗都打不开了 0 0
我的花呗额度***，为什么借呗就不行 花呗的额度怎么增加 0 0
我手机坏了所以借呗给逾期你能不能给处理一下 手机坏了借呗逾期一天咋办 0 0
我使用花呗冲话费充错号码了怎么办 我用花呗，充值话费，号码冲错了，能返还吗 0 0
使用多了支付宝，借呗额度恢复的快吗 借呗额度是实时恢复吗 0 0
花呗分完期，在分可以吗 我的花呗为什么不可以分***期付款 0 0
花呗第一次开通额度会是多少 花呗的最高额度多少 0 0
花呗我也没有逾期还款 花呗逾期没还清 0 0
花呗在那些地方可以线下支付 我的花呗怎么线下消费不可以 0 0
签约商户花呗分期 花呗商家店 0 0
借呗多久不用 就自动关闭了 现在我申请了蚂蚁借呗，我不多久不用他自动会关掉 0 1
同一个名下账号可以用两个花呗吗 为什么花呗一天只可以用*** 0 0
借呗已经逾期一天 借呗逾期了一天会不会信用降低 0 0
我的花呗都是按时处理账务为什么不能用了 怎么我的花呗不可以用了 0 0
花呗租衣服押金 花呗租东西要押金吗 0 0
不小心逾期，但是已全额还款，能激活借呗额度吗 借呗当天下午还款，算不算逾期 0 0
借呗怎么中午还没自动还款 为什么没有自动还款给蚂蚁借呗 1 0
花呗怎样设置最低还款 花呗最低还款后也可以一次结清吗 0 0
开通的花呗 为什么还开通不了花呗 0 0
怎么让花呗不能开通 我这个支付宝花呗关闭了，另外一个支付宝怎么开通不了 0 0
我用花呗充电话费为什么我微信钱包里的钱少了 花呗里的钱能当红包用吗 0 0
怎么查询商家收钱码花呗支付上限 支付宝收款码能收花呗钱吗 0 0
我上个月还款的花呗，怎么这个月的帐单还要还 为什么我已经付款了，花呗还要还款 0 0
开通当面付，把我花呗冻结了 怎么把我花呗冻结了 0 0
花呗免密码 花呗免密支付 1 1
```
最后相关的api可以去查阅：https://github.com/UKPLab/sentence-transformers