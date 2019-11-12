
from tqdm import tqdm
import os, json, codecs
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

data_base_dir = "/search/odin/liuyouyuan/pyproject/data/weibo_source"
article_file = "train_art.txt"
abstract_file = "train_abs.txt"
vocab_path = "./data/weibo_vocab.json"

# bert 相关
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

min_count = 0
max_input_len = 256
max_output_len = 32
batch_size = 16
steps_per_epoch = 1000  # 37500.0
epochs = 1000

print(f"args:{min_count}-{max_input_len}-{max_output_len}-{batch_size}")
model_name = './model/weibo_model_{}.weights'


def read_text(art_file, abs_file):
    with open(art_file, "r") as art_f, open(abs_file, "r") as abs_f:
        for t, s in zip(art_f, abs_f):
            if len(s) <= max_output_len:
                yield t[:max_input_len], s

def build_vocab_json(vocab_json, data):
    if os.path.exists(vocab_json):
        chars_dic = json.load(open(vocab_json, encoding='utf-8'))
    else:
        chars_dic = {}
        for tup in tqdm(data, desc='构建字表中'):
            for tex in tup:
                for c in tex:
                    chars_dic[c] = chars_dic.get(c, 0) + 1
        chars_dic = [(i, j) for i, j in chars_dic.items() if j >= min_count]
        chars_dic = sorted(chars_dic, key=lambda c: - c[1])
        chars_dic = [c[0] for c in chars_dic]
        json.dump(
            chars_dic,
            codecs.open(vocab_json, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )
    print("构建字表成功：", vocab_json) 
    return chars_dic


def padding(x):
    """
    padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator(tokenizer, art_abs_data):
    """构造输入数据流"""
    while True:
        X, Y = [], []
        for art, abstract in art_abs_data:
            x, y = tokenizer.encode(art, abstract)
            X.append(x)
            Y.append(y)
            if len(X) == batch_size:
                X = padding(X)
                Y = padding(Y)
                yield [X, Y], None
                X, Y = [], []

def gen_sent(model, tokenizer, s, topk=2):
    """
    beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    for i in range(max_output_len):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        # 直接忽略[PAD], [UNK], [CLS]
        _probas = model.predict([_target_ids, _segment_ids])[:, -1, 3:]
        # 取对数，方便计算
        _log_probas = np.log(_probas + 1e-6)
        # 每一项选出topk
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]
        for j, k in enumerate(_topk_arg):
            target_ids[j].append(_candidate_ids[k][-1])
            target_scores[j] = _candidate_scores[k]
        ends = [j for j, k in enumerate(target_ids) if k[-1] == 3]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def show(model, tokenizer, s_list):
    for s in s_list:
        print('生成摘要:', gen_sent(model, tokenizer, s))
    print()


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            if epoch % 5 == 0:
                print("Epoch: {} | Loss: {}".format(epoch, logs['loss']))
                model.save_weights(model_name.format(epoch))
        # 演示效果
        show(model, tokenizer, s_list)


s1 = '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时 就医 。'
s2 = '近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。但后据证实，章子怡的“大肚照”只是影片宣传的噱头。后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期'
s3 = '中国数字经济蓬勃发展，正成为创新经济发展方式的新引擎。信息技术为制造业、商业等各行各业有效赋能，推动国家不断前进。全国工商联副主席、正泰集团董事长南存辉说，公司深耕制造业数字化转型升级，努力探索物联网技术与智慧能源的深度融合之路。杭州大搜车集团首席执行官姚军红说，当前传统汽车行业进入深度调整期，流通市场急需变革，以降低车辆消费门槛，提振消费市场信心。大搜车采用人工智能、大数据等技术，建立开放平台，精准对接传统工厂、商家和客户，为消费者带来切实利益。旷视科技联合创始人兼首席执行官印奇说，新一轮互联网科技革命和产业变革加速演进，让从业者深切感受到机遇与挑战并存。印奇表示，旷视始终坚持自主创新，已成为全球为数不多的有自主研发深度学习框架的公司之一，将努力帮助中国企业在人工智能时代来临时，成为引领世界的力量。'
s_list = [s1,s2,s3]


# 从文件读取文章与参考摘要
art_file = os.path.join(data_base_dir, article_file)
abs_file = os.path.join(data_base_dir, abstract_file)
data = read_text(art_file, abs_file)

# 构建自己的字表
vocab_chars_dic = build_vocab_json(vocab_path,data)
# 读取bert词典
_token_dict = load_vocab(dict_path)
# 构建新的token_dict 用于构建Tokenizer
# keep_words是在bert中保留的字表
token_dict, keep_words = {}, []
for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])
for c in vocab_chars_dic:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])       
# 建立分词器
tokenizer = SimpleTokenizer(token_dict)

# 定义模型
model = load_pretrained_model(
    config_path,
    checkpoint_path,
    seq2seq=True,
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
# 目标tokens
y_in = model.input[0][:, 1:]
y_mask = model.input[1][:, 1:]
# 预测tokens，预测与目标错开一位
y = model.output[:, :-1]
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


if __name__ == "__main__":
    # 训练
    evaluator = Evaluate()
    model.fit_generator(
         data_generator(tokenizer, data),
         steps_per_epoch=steps_per_epoch,
         epochs=epochs,
         callbacks=[evaluator]
     )
    # 预测效果
    #model.load_weights(model_name)
    #show(model, tokenizer, s_list)

