'''
    This script is used to parse the data from BGL2 logs. 
    用于对数据处理，然后使用注意力机制实现对日志序列嵌入向量的获取  
'''
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import os
from model.semantic_embed import SemanticEmbed
import Drain
import ast

input_dir = '../log_data/BGL2'
output_dir = '../output/BGL2'
def preprocess_data(df, mode):
    # 这个函数中使用窗口大小对日志进行划分
    x_data, y_data = [], []
    if len(df) % 20 != 0:
        print('error length')
        return

    semantic_embed = SemanticEmbed()

    num_windows = int(len(df) / 20)
    for i in tqdm(range(num_windows)):
        df_blk = df[i*20:i*20+20]
        # 增加一个参数增强方法
        batch_vectors = np.array(df_blk['Vector'].tolist())
        template_vectors = torch.from_numpy(batch_vectors)
        df_blk['ParameterList'] = df_blk['ParameterList'].apply(ast.literal_eval)
        param_list = df_blk['ParameterList'].tolist()
        param_positions = df_blk['ParameterPositions'].tolist()

        # 获取语义增强后的嵌入向量
        log_embeddings = semantic_embed(template_vectors, param_list, param_positions)
        # 使用增强后的向量替代原始向量
        x_data.append(log_embeddings.detach().cpu().numpy())

        # 清理内存
        del template_vectors, log_embeddings, batch_vectors
        torch.cuda.empty_cache()

        labels = df_blk["Label"].tolist()
        if labels == ['-'] * 20: # 只有20条日志全部为正常时，日志序列才被认为是正常的
            y = [1, 0]
        else:
            y = [0, 1]
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    np.save(f'{output_dir}/{mode}_x.npy', x_data)
    np.save(f'{output_dir}/{mode}_y.npy', y_data)
    return len(x_data)

def split_data():
    log_name = 'bgl2'
    if not os.path.exists(f"{output_dir}/bgl2.log_structured.csv"):
        print("Parsing data...")
        log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'

        bgl_regex = [
            r'core\.\d+',
            r'(?:\/[\*\w\.-]+)+',  # path
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'0x[0-9a-f]+(?: [0-9a-f]{8})*',  # hex
            r'[0-9a-f]{8}(?: [0-9a-f]{8})*',
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5
        depth = 4

        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=bgl_regex)
        parser.parse(log_name + '.log')
    else:
        print("Data already parsed.")
    num_worker = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用sentence-transformer对日志模版进行编码
    model = SentenceTransformer(
        'distilbert-base-nli-mean-tokens', device=device)
    
    # 加载数据
    df_template = pd.read_csv(f"{output_dir}/{log_name}.log_templates.csv")
    df_structured = pd.read_csv(f"{output_dir}/{log_name}.log_structured.csv")

    # 计算所有已知模版向量
    print("Calculating all known templates...")
    embeddings = model.encode(df_template['EventTemplate'].tolist())
    df_template['Vector'] = list(embeddings)
    template_dict = df_template.set_index('EventTemplate')['Vector'].to_dict()

    # 每个日志都获得模版向量
    vectors = []
    for idx, template in enumerate(df_structured['EventTemplate']):
        try:
            vectors.append(template_dict[template])
        except KeyError:
            vectors.append(model.encode(template))
    df_structured['Vector'] = vectors
    print('done')
    df_structured.drop(
        columns = ['Date', 'Node', 'Time', 'NodeRepeat', 'Type', 'Component', 'Level']
    )

    # 添加一个参数增强
    print('Adding parameter augmentation...')
    parameter_positions = []
    contents = df_structured['Content'].tolist()

    a = 0
    paramlists = df_structured['ParameterList'].tolist()
    for idx, (content, paramlist) in tqdm(enumerate(zip(contents, paramlists)), desc='Get Parameter Positions', total=len(contents)):
        positions = []
        params = []
        if not paramlist:
            parameter_positions.append([])
            df_structured.at[idx,'ParameterList'] = str([])
            continue

        # 对参数再进行一次分词处理
        processed_params = []
        for param in paramlist:
            param_parts = param.strip().split()
            processed_params.append(param_parts)

        df_structured.at[idx, 'ParameterList'] = str(processed_params)

        words = content.split()

        # 获取参数的原始位置
        for part in processed_params:
            try:
                found = False
                min_position = -1
                last_position = positions[-1] if positions else -1
                for i, word in enumerate(words):
                    if i <= last_position:
                        continue
                    if part == word:
                        positions.append(i)
                        params.append(part)
                        found = True
                        break
                    elif part in word:
                        if (len(word) - len(part) <= 2 * len(word)) or (word.startswith(part) and word.endswith(part)):
                            positions.append(i)
                            params.append(part)
                            found = True
                            break
                if not found:
                    continue
            except:
                continue
        parameter_positions.append(positions)
    df_structured['ParameterPositions'] = parameter_positions
    print('参数位置提取完成！')

    num_windows = len(df_structured) // 20
    df_structured = df_structured.iloc[:num_windows*20]
    
    # 设置训练集大小
    training_windows = (num_windows // 5) * 4
    training_size = training_windows * 20
    
    # 直接在DataFrame上设置Usage列
    df_structured['Usage'] = 'testing'
    df_structured.loc[:training_size-1, 'Usage'] = 'training'

    # 根据Usage列分割数据
    df_test = df_structured[df_structured['Usage'] == 'testing']
    df_train = df_structured[df_structured['Usage'] == 'training']

    train_size = preprocess_data(df_train, 'train')
    test_size = preprocess_data(df_test, 'test')
    print(f'train size: {train_size}, test size: {test_size}')

if __name__ == '__main__':
    split_data()