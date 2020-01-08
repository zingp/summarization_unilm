from tqdm import tqdm
import os, json, codecs
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd


class Config(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'               # 使用GPU编号
        self.storge_path = "./storage"                         # 数据、模型存储路径
        self.pretrained_model_path = os.path.join(
             self.storge_path, "pretrain_model")               # bert 预训练模型存储路径
        # bert 预训练模型路径 chinese_L-12_H-768_A-12
        self.bert_config = os.path.join(
            self.pretrained_model_path, "bert_config.json")
        self.bert_checkpoint = os.path.join(
            self.pretrained_model_path, "bert_model.ckpt")
        self.bert_vocab= os.path.join(
            self.pretrained_model_path, "vocab.txt")

        self.data_path = os.path.join(self.storge_path, "data")  # 存储训练数据
        self.train_file = os.path.join(self.data_path, "train.txt")  # 训练数据
        self.dev_file = os.path.join(self.data_path, "dev.txt")
        self.vocab_file = os.path.join(
            self.data_path, "custom_vocab.txt")                  # 自定义词表文件

        self.model_path = os.path.join(
            self.storge_path, "models")                         # 存储训练得到的模型
        self.model_name = os.path.join(
            self.model_path, "unilm_model.bin")                 # 模型名称
        for d in [self.storge_path,self.data_path,self.model_path]:
            if not os.path.isdir(d):
                os.mkdir(d)

        # 训练参数
        # config_path = './data/custom_vocab.txt'
        self.min_count = 0          # 最低词频
        self.max_input_len = 256    # 输入最大序列长度
        self.max_output_len = 32    # 生成最大序列长度
        self.batch_size = 16        # 批次大小
        self.steps_per_epoch = 1000  # 迭代次数？
        self.epochs = 10     


def read_text(config):
    """从文件读取文章、摘要"""
    with open(config.train_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line) 
            article = text["article"]
            summary = text["summarization"]
            if len(summary) <= config.max_output_len:
                yield article[:config.max_input_len], summary


def build_custom_vocab(config):
    """构建自定义词典"""
    if os.path.exists(config.vocab_file):
        print("loading custom vocab form local ....")
        chars = json.load(open(config.vocab_file, encoding='utf-8'))
    else:
        chars = {}
        for a in tqdm(read_text(config), desc='构建字表中'):
            for b in a:
                for w in b:
                    chars[w] = chars.get(w, 0) + 1
        chars = [(i, j) for i, j in chars.items() if j >= config.min_count]
        # chars = [(i, j) for i, j in chars.items()]
        chars = sorted(chars, key=lambda c: - c[1])
        chars = [c[0] for c in chars]
        json.dump(
            chars,
            codecs.open(config.vocab_file, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )
    return chars


def build_vocab(config):
    """将自定义词典加入bert的词典中"""
    # 读取词典
    _token_dict = load_vocab(config.bert_vocab)
    # keep_words是在bert中保留的字表
    token_dict, keep_words = {}, []

    for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])
    chars = build_custom_vocab(config)
    for c in chars:
        if c in _token_dict:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])
    return token_dict, keep_words


def padding(x):
    """padding至batch内的最大长度"""
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator(config, tokenizer):
    """生成训练数据流"""
    while True:
        X, S = [], []
        for a, b in read_text(config):
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            if len(X) == config.batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []


def creat_model(config, keep_words):
    """构建模型"""
    model = load_pretrained_model(
        config.bert_config,
        config.bert_checkpoint,
        seq2seq=True,
        keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    )

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

    return model


def gen_sent(config, tokenizer, model, s, topk=2):
    """
    beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:config.max_input_len])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    for i in range(config.max_output_len):
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


def show(config, tokenizer, model):
    s1 = '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时 就医 。'
    s2 = '近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。但后据证实，章子怡的“大肚照”只是影片宣传的噱头。后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期'
    s3 = '中国数字经济蓬勃发展，正成为创新经济发展方式的新引擎。信息技术为制造业、商业等各行各业有效赋能，推动国家不断前进。全国工商联副主席、正泰集团董事长南存辉说，公司深耕制造业数字化转型升级，努力探索物联网技术与智慧能源的深度融合之路。杭州大搜车集团首席执行官姚军红说，当前传统汽车行业进入深度调整期，流通市场急需变革，以降低车辆消费门槛，提振消费市场信心。大搜车采用人工智能、大数据等技术，建立开放平台，精准对接传统工厂、商家和客户，为消费者带来切实利益。旷视科技联合创始人兼首席执行官印奇说，新一轮互联网科技革命和产业变革加速演进，让从业者深切感受到机遇与挑战并存。印奇表示，旷视始终坚持自主创新，已成为全球为数不多的有自主研发深度学习框架的公司之一，将努力帮助中国企业在人工智能时代来临时，成为引领世界的力量。'
    for s in [s1, s2, s3]:
        print('生成摘要:', gen_sent(config, tokenizer, model, s))
    print()


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, config, tokenizer, model, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(config.model_name)
        # 演示效果
        show(config, tokenizer, model)


if __name__ == '__main__':

    config = Config()
    token_dict, keep_words = build_vocab(config)
    # 建立分词器
    tokenizer = SimpleTokenizer(token_dict)
    # 构建模型
    model = creat_model(config, keep_words)
    model.summary()

    evaluator = Evaluate()
    # 开始训练
    model.fit_generator(
         data_generator(config, tokenizer),
         steps_per_epoch=config.steps_per_epoch,
         epochs=config.epochs,
         callbacks=[evaluator]
     )
    
    # 预测
    #model.load_weights(model_name)
    #show()
     
