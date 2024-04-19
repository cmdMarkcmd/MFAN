import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import gc
from time import *
import re
from transformers import BertConfig ,BertModel,BertTokenizer

def data_process():
    print('data_process: Start')
    pre = os.getcwd() + "/dataset/weibo/weibo_files"
    X_train_tid, _, _,_,_= pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, _, _ = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, _, _ = pickle.load(open(pre + "/test.pkl", 'rb'))
    #mid是microblog id 既有评论也有微博甚至还包含了uid
    with open(pre+ '/new_id_dic.json', 'r') as f:
        mid2newid = json.load(f)#mid->newid mid:Ccli05cxs类似的乱码 newid：从1开始编号的整数
        newid2mid = dict(zip(mid2newid.values(),mid2newid.keys()))#newid->mid
    mid2num = {} 
    text_list = [] 
    id_list = []
    path = os.getcwd()+'/dataset/weibo/weibocontentwithimage/original-microblog/'
    for file in os.listdir(path):
        mid2num[file.split('_')[-2]] = file.split('_')[0]#存储对应关系 mid对应的图片id
        id_list.append(file.split('_')[0])#存储图片id列表
        with open(path+file, 'r',encoding='utf-8') as f:
            reader = json.load(f)
            text_list.append(remove_links(reader["text"]))#这里插入了微博
     
    index = 0
    num2index = {}#imid->index
    for i in id_list:
        num2index[i] = index#图片id num对应的文本id index
        index+=1
    
    text_weight,text_weight_1,text_weight_2,text_weight_3\
       = bert_process_part(text_list)
    gc.collect()#garbage collector
    torch.cuda.empty_cache()

    newid2num = {}#new_id->imid
    newid2index= {}#new_id->index(post)
    mid2index = {}#mid(post)->index
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]#存储num 找到newid对应的图片
        mid2index[newid2mid[id]]= newid2index[id] = num2index[newid2num[id]]#num取对应index和newid组成键值对 找到mid和newid对应的文本
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
        mid2index[newid2mid[id]]= newid2index[id] = num2index[newid2num[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
        mid2index[newid2mid[id]]= newid2index[id] = num2index[newid2num[id]]
   
    
    #mid2comment={}
    mid2commentid={}#mid也可对应的评论
    comment_list=[]
    comment_cnt = 0
    with open(pre+'/comment_content.json','r',encoding = 'utf-8') as f:
        id_comment_map = json.load(f)
        for mid,text in id_comment_map.items():#mid和对应的评论
            nolinks_text = remove_links(text)#去掉评论中的连接
            #mid2comment[mid] = nolinks_text
            mid2commentid[mid] = comment_cnt
            comment_cnt += 1
            comment_list.append(nolinks_text)

    comment, comment_1, comment_2, comment_3\
        = bert_process_part(comment_list) #一次性处理完比一个一个处理快的多
    gc.collect()#garbage collector
    torch.cuda.empty_cache()

    uid2mids={} #每个用户uid对应的微博mid list
    with open(pre+'/user_tweet.json','r',encoding='utf-8') as f:
        input = json.load(f)
        for uid,midlist in input.items():
            uid2mids[uid] = midlist

    num_node = len(newid2mid)#涵盖所有下标到节点id的转换关系
    
    node_embedding_matrix = np.random.uniform(-0.25, 0.25, (num_node, 768)) #形状为 (num_node, 768) 的随机初始化的矩阵
    for i in range(num_node):
        id = newid2mid[i]
        if id in mid2index:#如果id是post_mid，直接读取嵌入
            index = mid2index[id] #微博在第42行已经处理
            node_embedding_matrix[i,:] = torch.mean( #把[1,768]的向量按第一维连接变成[4,768]然后按第一维取平均变成[1,768]
                  torch.cat(( text_weight[index]
                           ,text_weight_1[index]
                           ,text_weight_2[index]
                           ,text_weight_3[index])
                           ,dim=0)
                           ,dim=0
                           ,keepdim=False)
        
        elif id in mid2commentid:#如果id是comment_mid，处理后嵌入
            #comment_text = mid2comment[id]
            comment_id = mid2commentid[id]
            # comment,comment_1,comment_2,comment_3=\
            #     bert_process_part([comment_text])
            node_embedding_matrix[i,:]=torch.mean(
                  torch.cat((comment[comment_id],
                             comment_1[comment_id],
                             comment_2[comment_id],
                             comment_3[comment_id])
                           ,dim=0)
                           ,dim=0
                           ,keepdim=False)
        
    for i in range(num_node):#在源帖子节点和评论节点都有数值之后
        id = newid2mid[i]
        if id in uid2mids:
            posts = uid2mids[id]
            # print("i = {}, id = {} , post = {}".format(i, id, posts))
            count = 0
            embedding = 0.0
            for post in posts:#用户uid相关的mid
                count+=1
                j = mid2newid[post]#得到相应的下标
                embedding+=node_embedding_matrix[j]
            if count>0:
                 embedding/=count
                 node_embedding_matrix[i,:] = embedding#用户节点就是其发表的帖子和评论的加权平均
    
    torch.save(text_weight,pre+'/weight.pt')
    torch.save(text_weight_1,pre+'/weight_1.pt')
    torch.save(text_weight_2,pre+'/weight_2.pt')
    torch.save(text_weight_3,pre+'/weight_3.pt')
    save_path = pre + "/node_embedding.pkl"
    if os.path.exists(save_path):
        os.remove(save_path)
    pickle.dump([node_embedding_matrix],
                open(save_path, 'wb')) #将所有节点的嵌入保存在node_embedding.pkl
    

    print('data_process: Done')
    

def bert_process_part(text_list):
    length = len(text_list)
    modal_path =os.getcwd() + "/bert-base-chinese/"
    config = BertConfig.from_pretrained(modal_path, output_hidden_states = True)
    assert config.output_hidden_states == True
    bert_modal = BertModel.from_pretrained(modal_path,config = config).cuda() #获得预训练参数和config
    bert_tokenizer = BertTokenizer.from_pretrained(modal_path)#获得分词器
    
    lines=[]
    for line in text_list:
        lines.append(bert_tokenizer.encode(line,max_length = 200\
                                               ,padding = 'max_length'\
                                               ,truncation = 'longest_first'))
    bert_modal.eval()#非训练模式
    outputs=[]
    layers_1 = []
    layers_2 = []
    layers_3 = []
    num = 0
    batch_size = 50
    print("total batches: {}".format(length / batch_size))
    #这样处理之后在文本训练相关模型中，不需要x_train等，只留一个X_train_tid即可
    for i in range(int(length / batch_size)):#0-19分批次处理，防止显存爆炸
        print("processing text batch: {}".format(i))
        num+=1
        batch_lines=lines[i * batch_size : (i+1) * batch_size]#截取50行作为1batch
        input_ids=torch.tensor(batch_lines,dtype = torch.long).cuda()
        mask = torch.where(input_ids!=0,torch.ones_like(input_ids),torch.zeros_like(input_ids)).cuda()
        this_put = bert_modal(input_ids,attention_mask=mask)[0].detach().cpu()
        that_put = torch.stack(bert_modal(input_ids,attention_mask=mask)[2]).detach().cpu()
        hidden_layer_1=torch.zeros_like(this_put)
        hidden_layer_2=torch.zeros_like(this_put)
        hidden_layer_3=torch.zeros_like(this_put)
        for i in range(12):#4为组读取隐藏层输出，并且各自加和
            if i<4:
                hidden_layer_1+=that_put[i]
            elif i<8:
                hidden_layer_2+=that_put[i]
            else:
                hidden_layer_3+=that_put[i]
        outputs.append(this_put)
        layers_1.append(hidden_layer_1)
        layers_2.append(hidden_layer_2)
        layers_3.append(hidden_layer_3)
        gc.collect()
        torch.cuda.empty_cache()
    print("====this part end====")
    batch_lines=lines[num * batch_size:]
    input_ids=torch.tensor(batch_lines,dtype = torch.long).cuda()
    mask = torch.where(input_ids!=0,torch.ones_like(input_ids),torch.zeros_like(input_ids))
    this_put = bert_modal(input_ids,attention_mask=mask)[0].detach().cpu()
    that_put = torch.stack(bert_modal(input_ids,attention_mask=mask)[2]).detach().cpu()
    
    hidden_layer_1=torch.zeros_like(this_put)
    hidden_layer_2=torch.zeros_like(this_put)
    hidden_layer_3=torch.zeros_like(this_put)
    for i in range(12):
        if i<4:
            hidden_layer_1+=that_put[i]
        elif i<8:
            hidden_layer_2+=that_put[i]
        else:
            hidden_layer_3+=that_put[i]
    outputs.append(this_put)
    layers_1.append(hidden_layer_1)
    layers_2.append(hidden_layer_2)
    layers_3.append(hidden_layer_3)
    outputs = torch.cat(outputs,dim=0)
    layers_1 = torch.cat(layers_1,dim=0)
    layers_2 = torch.cat(layers_2,dim=0)
    layers_3 = torch.cat(layers_3,dim=0)
    outputs = torch.mean(outputs,dim=1,keepdim=False)
    layers_1 = torch.mean(layers_1,dim=1,keepdim=False)
    layers_2 = torch.mean(layers_2,dim=1,keepdim=False)
    layers_3 = torch.mean(layers_3,dim=1,keepdim=False)#[size,768]
    '''trans = outputs.shape
    outputs = outputs.view(trans[0],trans[1]*trans[2])
    layers_1 = layers_1.view(trans[0],trans[1]*trans[2])
    layers_2 = layers_2.view(trans[0],trans[1]*trans[2])
    layers_3 = layers_3.view(trans[0],trans[1]*trans[2])'''
    
   
    
    
    #准备用这四个张量搞四个可训练的嵌入层
    return outputs,layers_1,layers_2,layers_3



def remove_links(string):
    # 使用正则表达式匹配链接并替换为空字符串

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    clean_text = re.sub(r"\s{2,}", " ", string)
    return clean_text

data_process()