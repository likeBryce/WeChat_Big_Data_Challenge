# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
# # 这行代码是 TensorFlow 1.x 兼容模式导入：
# TensorFlow 2.x 默认使用 Eager Execution（即时执行）
# TensorFlow 1.x 使用 Graph Execution（图执行）
# 这行代码让 TensorFlow 2.x 环境能够运行 1.x 风格的代码
import tensorflow.compat.v1 as tf

from tensorflow import feature_column as fc
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from evaluation import uAUC, compute_weighted_score

# 这段代码使用了 TensorFlow 的命令行参数解析系统。让我详细解释每一部分：
# 这是 TensorFlow 1.x 风格的命令行参数定义，用于配置模型的超参数和路径。
flags = tf.app.flags # TensorFlow 的参数解析模块
FLAGS = flags.FLAGS # 全局对象，用于存储和访问所有定义的参数

# 名称: model_checkpoint_dir  默认值: './data/model'  描述: 'model dir'
flags.DEFINE_string('model_checkpoint_dir', './data/model', 'model dir') # 模型检查点路径: 定义模型保存和加载的目录路径
flags.DEFINE_string('root_path', './data/', 'data dir') # 数据根路径: 定义数据文件的根目录
flags.DEFINE_integer('batch_size', 128, 'batch_size') # 批量大小: 定义训练时每个批次的样本数量
flags.DEFINE_integer('embed_dim', 10, 'embed_dim') # 嵌入维度: 定义嵌入向量的维度大小
flags.DEFINE_float('learning_rate', 0.1, 'learning_rate') # 学习率: 定义优化器的学习率
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg') # L2 正则化: 定义嵌入层的 L2 正则化系数

# 在代码中访问示例:
# print(f"批量大小: {FLAGS.batch_size}")
# print(f"学习率: {FLAGS.learning_rate}")
# print(f"模型路径: {FLAGS.model_checkpoint_dir}")
SEED = 2021

class WideAndDeep(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, stage, action):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(WideAndDeep, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "favorite": 1, "forward": 1,
                                "comment": 1, "follow": 1}
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.stage = stage
        self.action = action
        tf.logging.set_verbosity(tf.logging.INFO)

    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_checkpoint_stage_dir = os.path.join(FLAGS.model_checkpoint_dir, stage, self.action)
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(model_checkpoint_stage_dir)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1) # 数值稳定性的常数（通常很小，这里设为1可能是个调优结果）
        config = tf.estimator.RunConfig(model_dir=model_checkpoint_stage_dir, tf_random_seed=SEED)
        self.estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_checkpoint_stage_dir,
            linear_feature_columns=self.linear_feature_columns, # Linear部分：记忆（Memorization）- 处理稀疏特征, 学习特征组合, 示例：用户A经常点击作者B的内容
            dnn_feature_columns=self.dnn_feature_columns, # DNN部分：泛化（Generalization）- 处理稠密特征和嵌入向量, 学习特征深层交互, 示例：基于用户历史行为预测新内容偏好
            dnn_hidden_units=[32, 8], # 两个隐藏层，神经元数分别为32和8; # 第一层32神经元：学习复杂特征交互; 第二层8神经元：提取核心特征表示; 防止过拟合，提高泛化能力
            dnn_optimizer=optimizer, # 使用配置的Adam优化器
            config=config)
        """
        记忆的工作原理: 
        用户A + 视频B → 点击 (正样本)
        用户A + 视频B → 点击 (正样本)  
        用户A + 视频B → 点击 (正样本)
        Linear部分学习:
        权重[用户A_视频B] = +3.2  (高权重，表示强关联)

        泛化的工作原理:
        # DNN学习到的模式
        用户A喜欢视频X (在训练数据中)
        用户B的嵌入向量 ≈ 用户A的嵌入向量 (相似用户)
        视频Y的嵌入向量 ≈ 视频X的嵌入向量 (相似视频)

        # 泛化预测：用户B也可能喜欢视频Y
        即使用户B和视频Y从未在训练数据中共现过！

        # 可以尝试不同的学习率
        optimizers = {
        'high_lr': tf.train.AdamOptimizer(learning_rate=0.1),
        'medium_lr': tf.train.AdamOptimizer(learning_rate=0.01),
        'low_lr': tf.train.AdamOptimizer(learning_rate=0.001)}

        # 不同的网络结构实验
        architectures = {
            'small': [16, 4],      # 小网络
            'medium': [32, 8],     # 中等网络（当前使用）
            'large': [64, 16, 4],  # 大网络
            'deep': [128, 64, 32, 8]  # 更深网络}
        """

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        '''
        把DataFrame转为tensorflow dataset
        :param df: pandas dataframe. 
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        :param shuffle: Boolean. 
        :param batch_size: Int. Size of each batch
        :param num_epochs: Int. Epochs num
        :return: tf.data.Dataset object. 
        '''
        # print(df.shape)
        # print(df.columns)
        # print("batch_size: ", batch_size)
        # print("num_epochs: ", num_epochs)
        if stage != "submit": # 如果不是提交阶段（submit），则数据集包含特征和标签。
            label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else: # 如果是提交阶段，则数据集只包含特征（没有标签）。
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED) # 在训练时打乱数据顺序，防止模型学习到数据顺序
        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs) # 在训练阶段重复数据集多个epoch
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=FLAGS.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self):
        """
        训练单个行为的模型
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                      day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(df, self.stage, self.action, self.num_epochs_dict[self.action])
        )

    def evaluate(self):
        """
        评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                      day=STAGE_END_DAY[self.stage])
        evaluate_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    
    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
                                                                      day=STAGE_END_DAY[self.stage])
        submit_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time()-t)*1000.0/len(df)*2000.0
        return df[["userid", "feedid"]], logits, ts

    

def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def get_feature_columns():
    '''
    获取特征列
    '''
    dnn_feature_columns = list()
    linear_feature_columns = list()
    # DNN features
    user_cate = fc.categorical_column_with_hash_bucket("userid", 40000, tf.int64)
    feed_cate = fc.categorical_column_with_hash_bucket("feedid", 240000, tf.int64)
    author_cate = fc.categorical_column_with_hash_bucket("authorid", 40000, tf.int64)
    bgm_singer_cate = fc.categorical_column_with_hash_bucket("bgm_singer_id", 40000, tf.int64)
    bgm_song_cate = fc.categorical_column_with_hash_bucket("bgm_song_id", 60000, tf.int64)
    user_embedding = fc.embedding_column(user_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    feed_embedding = fc.embedding_column(feed_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    author_embedding = fc.embedding_column(author_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    bgm_singer_embedding = fc.embedding_column(bgm_singer_cate, FLAGS.embed_dim)
    bgm_song_embedding = fc.embedding_column(bgm_song_cate, FLAGS.embed_dim)
    dnn_feature_columns.append(user_embedding)
    dnn_feature_columns.append(feed_embedding)
    dnn_feature_columns.append(author_embedding)
    dnn_feature_columns.append(bgm_singer_embedding)
    dnn_feature_columns.append(bgm_song_embedding)
    # Linear features
    video_seconds = fc.numeric_column("videoplayseconds", default_value=0.0)
    device = fc.numeric_column("device", default_value=0.0)
    linear_feature_columns.append(video_seconds)
    linear_feature_columns.append(device)
    # 行为统计特征
    for b in FEA_COLUMN_LIST:
        feed_b = fc.numeric_column(b+"sum", default_value=0.0)
        linear_feature_columns.append(feed_b)
        user_b = fc.numeric_column(b+"sum_user", default_value=0.0)
        linear_feature_columns.append(user_b)
    return dnn_feature_columns, linear_feature_columns


def main(argv):
    t = time.time() 
    dnn_feature_columns, linear_feature_columns = get_feature_columns()
    stage = argv[1]
    print('Stage: %s'%stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    for action in ACTION_LIST:
        print("Action:", action)
        model = WideAndDeep(linear_feature_columns, dnn_feature_columns, stage, action)
        model.build_estimator()

        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)


    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(FLAGS.root_path, stage, file_name)
        print('Save to: %s'%submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    tf.app.run(main)
    