# coding: utf-8
import os
import time
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd

# 存储数据的根目录
DATASET_PATH = "/workspace/data"
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 2021

# 初赛待预测行为列表: 点赞、点击头像、收藏、转发
ACTION_LIST = ["read_comment"]
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 每个行为的负样本下采样比例(下采样后负样本数/原负样本数)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1, "comment": 0.1, "follow": 0.1, "favorite": 0.1}

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5, "favorite": 5}


def create_dir():
    """
    创建所需要的目录
    """
    # 创建data目录
    if not os.path.exists(DATASET_PATH):
        print('Create dir: %s'%DATASET_PATH)
        os.mkdir(DATASET_PATH)
    # data目录下需要创建的子目录
    need_dirs = ["offline_train", "online_train", "evaluate", "submit",
                 "feature", "model", "model/online_train", "model/offline_train"]
    for need_dir in need_dirs:
        need_dir = os.path.join(DATASET_PATH, need_dir)
        if not os.path.exists(need_dir):
            print('Create dir: %s'%need_dir)
            os.mkdir(need_dir)


def check_file():
    '''
    检查数据文件是否存在
    '''
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file


def statis_data():
    """
    统计特征最大，最小，均值
    """
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    # 设置 Pandas 显示选项，使得在打印 DataFrame 时显示所有列（不限制显示的列数）
    pd.set_option('display.max_columns', None)
    for path in paths:
        df = pd.read_csv(path)
        print(path + " statis: ")
        print(df.describe())
        print('Distinct count:')
        print(df.nunique())


def statis_feature(start_day=1, before_day=7, agg='sum'):
    """
    这是一个用于生成时间窗口统计特征的函数，在推荐系统中常用于构建用户和物品的历史行为特征。
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间窗口大小（统计过去多少天）
    :param agg: String. 统计方法
    """
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feature_dir = os.path.join(DATASET_PATH, "feature")
    for dim in ["userid", "feedid"]: # 分别对用户维度和内容维度生成统计特征
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, END_DAY-before_day+1): # [1, 9] 创建滑动时间窗口，为每个时间点生成历史统计特征
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_']) # 移除日期列: 因为我们要按用户/内容分组统计，日期信息不再需要。
            temp = temp.groupby([dim]).agg([agg]).reset_index() # 这里用'sum'是因为每个特征都是0-1变量
            """
            # 分组前：
            userid | click | like | share
            1001   | 1     | 0    | 0
            1001   | 1     | 1    | 0
            1001   | 1     | 0    | 1
            1002   | 1     | 0    | 0

            # 分组后 (按userid分组，agg='sum'):
            userid | click | like | share
            1001   | 3     | 1    | 1      # 1001用户过去7天的行为总和
            1002   | 1     | 0    | 0      # 1002用户过去7天的行为总和
            """

            # # 聚合前的列名结构：Pandas分组聚合操作产生的多级列名， userid列是分组列，没有聚合操作
            # [('userid', ''), ('click', 'sum'), ('like', 'sum'), ('share', 'sum')]
            # # 使用 map(''.join) 后：
            # ['userid', 'clicksum', 'likesum', 'sharesum']
            temp.columns = list(map(''.join, temp.columns.values)) # 将元组列名拼接成字符串
            temp["date_"] = start + before_day # 这个日期表示特征对应的目标日期：窗口 [1, 8) 的统计 → 用于预测第8天的行为
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        """
        # dim_feature 的最终结构：
        userid | clicksum | likesum | sharesum | date_
        1001   | 3        | 1       | 1        | 8      # 第1-7天统计，用于第8天
        1001   | 2        | 1       | 1        | 9      # 第2-8天统计，用于第9天
        1001   | 1        | 1       | 0        | 10     # 第3-9天统计，用于第10天
        ...    | ...      | ...     | ...      | ...
        1002   | 1        | 0       | 0        | 8      # 第1-7天统计，用于第8天
        1002   | 0        | 0       | 0        | 9      # 第2-8天统计，用于第9天
        """

        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)


def generate_sample(stage="offline_train"):
    """
    这个函数根据不同的阶段（训练/评估/提交）生成不同采样策略的数据样本。
    对负样本进行下采样，生成各个阶段所需样本
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    :return: List of sample df
    """
    day = STAGE_END_DAY[stage]
    if stage == "submit":
        sample_path = TEST_FILE
    else:
        sample_path = USER_ACTION
    stage_dir = os.path.join(DATASET_PATH, stage)
    df = pd.read_csv(sample_path)
    df_arr = []
    if stage == "evaluate":
        # 线下评估
        col = ["userid", "feedid", "date_", "device"] + ACTION_LIST
        df = df[df["date_"] == day][col] # 验证集只提取第13天的样本数据
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    elif stage == "submit":
        # 线上提交
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        df["date_"] = 15
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    else:
        # 线下/线上训练
        # 同行为取按时间最近的样本（修改为按userid、feedid和action去重，保留最近一条）
        # 数据丢失风险: 每次循环都在前一次去重结果基础上继续去重，可能导致数据过度过滤。
        # 修正：一次性去重，保留用户-物品最近交互
        df = df.drop_duplicates(subset=['userid', 'feedid'], keep='last')
        "存在待改进空间：1. 负样本下采样比例； 2. 重复数据处理方式"
        # for action in ACTION_LIST:
        #     df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last') # keep='last' 保留最后出现的重复记录
        # 负样本下采样
        for action in ACTION_LIST:
            # 步骤1：选择后 ACTION_DAY_NUM[action 时间窗口内的数据
            action_df = df[(df["date_"] <= day) & (df["date_"] >= day - ACTION_DAY_NUM[action] + 1)]
            # 步骤2：分离正负样本
            df_neg = action_df[action_df[action] == 0]
            df_pos = action_df[action_df[action] == 1]
            # 步骤3：负样本采样：这行代码从负样本中随机抽取指定比例的样本，用于解决类别不平衡问题。
            # frac: 指定采样比例 random_state: 设置随机种子，确保每次采样结果一致 replace=False:指定无放回抽样
            df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False) 
             # 步骤4：合并正负样本
            df_all = pd.concat([df_neg, df_pos])
            # 步骤5：选择需要的列并保存
            col = ["userid", "feedid", "date_", "device"] + [action]
            file_name = os.path.join(stage_dir, stage + "_" + action + "_" + str(day) + "_generate_sample.csv")
            print('Save to: %s'%file_name)
            df_all[col].to_csv(file_name, index=False)
            df_arr.append(df_all[col])
    return df_arr


def concat_sample(sample_arr, stage="offline_train"):
    """
    这是一个特征工程函数，主要作用是将原始的样本数据与物品特征、用户历史行为特征、物品历史行为特征进行合并，生成最终的特征数据集。
    :param sample_arr: List of sample df
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    """
    day = STAGE_END_DAY[stage]
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO) # 物品静态特征（作者、BGM、视频时长等）
    feed_info = feed_info.set_index('feedid') # 将原先的idx取消，将feedid列设置为 DataFrame 的索引（feedid从原先的列变为索引）
    
    # 基于userid统计的历史行为的次数
    user_date_feature_path = os.path.join(DATASET_PATH, "feature", "userid_feature.csv")
    user_date_feature = pd.read_csv(user_date_feature_path) # 用户每日行为统计特征
    user_date_feature = user_date_feature.set_index(["userid", "date_"])

    # 基于feedid统计的历史行为的次数
    feed_date_feature_path = os.path.join(DATASET_PATH, "feature", "feedid_feature.csv")
    feed_date_feature = pd.read_csv(feed_date_feature_path) # 物品每日行为统计特征
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"]) # 将feedid列和date_设置为 DataFrame 的索引

    for index, sample in enumerate(sample_arr):
        # 假设 sample_arr 包含3个DataFrame
        # sample_arr = [click_samples, like_samples, share_samples]
        # 第一次循环：index=0, sample=click_samples
        # 第二次循环：index=1, sample=like_samples  
        # 第三次循环：index=2, sample=share_samples

        # 定义所有样本共有的基础特征列
        features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id",
                    "videoplayseconds"]
        if stage == "evaluate": # 评估阶段：包含所有行为标签
            action = "all" # 标识当前处理所有行为
            features += ACTION_LIST # 将所有的行为标签列添加到特征列表中
        elif stage == "submit": # 提交阶段：不包含行为标签
            action = "all" # 标识当前处理所有行为
        else: # 训练阶段：只包含当前行为的标签
            action = ACTION_LIST[index]
            features += [action]

        print("action: ", action)
        # 1. 合并物品静态特征: 左连接效果：保留sample中的所有行;只在feed_info中存在的feedid才会合并特征;如果feedid在feed_info中不存在，对应特征列为NaN
        sample = sample.join(feed_info, on="feedid", how="left", rsuffix="_feed") # rsuffix="_feed": 当列名冲突时，为右侧DataFrame的列添加后缀_feed
        # 2. 合并物品历史行为特征
        sample = sample.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
        # 3. 合并用户历史行为特征
        sample = sample.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

        # 1. 填充缺失值
        feed_feature_col = [b+"sum" for b in FEA_COLUMN_LIST]
        user_feature_col = [b+"sum_user" for b in FEA_COLUMN_LIST]
        sample[feed_feature_col] = sample[feed_feature_col].fillna(0.0)
        sample[user_feature_col] = sample[user_feature_col].fillna(0.0)

        # 2. 对数变换（平滑处理）: 为什么要做对数变换：减少数值的偏态分布; 使特征更符合正态分布; 降低异常值的影响
        sample[feed_feature_col] = np.log(sample[feed_feature_col] + 1.0)
        sample[user_feature_col] = np.log(sample[user_feature_col] + 1.0)
        features += feed_feature_col
        features += user_feature_col

        # 3. 类别特征处理
        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # +1: 将ID类特征的0值保留给未知类别
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0) # 视频时长也做对数变换

        # 4. 类型转换
        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

        """
        假设输入数据： 
        sample (来自generate_sample函数)
        userid | feedid | date_ | device | click
        1      | 100    | 14    | ios    | 1

        经过特征合并后：
        # 合并feed_info后的sample
        userid | feedid | date_ | device | click | authorid | bgm_song_id | bgm_singer_id | videoplayseconds
        1      | 100    | 14    | ios    | 1     | 50       | 200         | 300           | 120

        # 合并feed_date_feature后的sample  
        userid | feedid | ... | clicksum | likesum | sharesum | ...
        1      | 100    | ... | 150      | 30      | 5        | ...

        # 合并user_date_feature后的sample
        userid | feedid | ... | clicksum_user | likesum_user | sharesum_user | ...
        1      | 100    | ... | 500           | 100          | 20            | ...

        """
        file_name = os.path.join(DATASET_PATH, stage, stage + "_" + action + "_" + str(day) + "_concate_sample.csv")
        print('Save to: %s'%file_name)
        sample[features].to_csv(file_name, index=False)


def main():
    t = time.time()
    statis_data()
    logger.info('Create dir and check file')
    create_dir() # 创建所需要的目录
    flag, not_exists_file = check_file()
    if not flag:
        print("请检查目录中是否存在下列文件: ", ",".join(not_exists_file))
        return
    logger.info('Generate statistic feature')
    statis_feature()
    for stage in STAGE_END_DAY:
        logger.info("Stage: %s"%stage)
        logger.info('Generate sample')
        sample_arr = generate_sample(stage)
        logger.info('Concat sample with feature')
        concat_sample(sample_arr, stage)
    print('Time cost: %.2f s'%(time.time()-t)) # 统计运行总时间


if __name__ == "__main__":
    main()
    