# encoding = utf-8
# 华为云项目，用雪瑶的数据集finetune

import os
import sys
#sys.path.append(os.path.abspath(".") + "/../")
sys.path.append(os.path.abspath("."))
#sys.path.append(os.path.abspath("."))
import random
import time
from datetime import timedelta
import torch
import clip.clip as clip
from clip.model import convert_models_to_fp32, convert_models_to_mix
from PIL import Image
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import random
import json
_DEBUG = False


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'CLIP-concept-finetune'
        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = './save/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备 

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        #self.num_epochs = 10                                             # epoch数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32
                                                   # mini-batch大小
        # self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-6
        # 学习率
        # self.bert_path = '/content/drive/Shared drives/A/data/pre_training/bert_pretain'  # 预训练模型
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # self.hidden_size = 768
        self.lr_scheduler = "cosine"                                    # 学习率衰减的方式（decay the learning rate using a cosine schedule)
        self.k_fold = 10

def flat_accuracy(preds, labels):
    #print (preds)
    #print (labels)
    return np.sum(preds == labels) / len(labels)
        
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    print ("start")
    model.train()  # model.train()将启用BatchNormalization和Dropout，相应的，model.eval()则不启用BatchNormalization和Dropout
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gain']
    #设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    #for param in model.parameters():
    #    param.requires_grad = False
    #for param in model.transformer.resblocks[10:12].parameters():
    #    param.requires_grad = True

    # Parameters:
    # 总的训练样本数
    total_steps = len(train_iter) * config.num_epochs
    # 创建学习率调度器
    #optimizer = AdamW(model.parameters(), lr = config.learning_rate)  # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(optimizer_grouped_parameters, lr = config.learning_rate)  # To reproduce BertAdam specific behavior set correct_bias=False
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)  # PyTorch scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs, eta_min = 0, last_epoch = -1)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        model.train()
        for i, batch in enumerate(train_iter):
            images, text = batch
            model.zero_grad()
            logits_per_image, logits_per_text = model(images.to(config.device), text.to(config.device))
            #print ("logits_per_image=", logits_per_image)
            #print ("logits_per_text=", logits_per_text)
            labels = torch.arange(logits_per_image.shape[0], device = config.device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            #print ("loss=", loss)

            # 反向传播
            optimizer.zero_grad()
            #model.zero_grad() # 等价于optimizer.zero_grad()，都是把梯度的参数置于0
            loss.backward()
            #for p,q in zip(model.named_parameters(), model.parameters()):
            #    if p[0].find("logit") > -1:     # 'logit_scale'
            #        print (p)
            #        print (q.grad.norm())

            # # 优化器
            # 优化的时候改为float32，forword的时候改为float16
            convert_models_to_fp32(model) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 100.0)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            model.after_step()
            convert_models_to_mix(model)
            
            ## validation

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # img_label_idx和text_label_idx拼接在一起计算准确率
                image_label = labels.detach().cpu().numpy()
                image_predict = torch.max(logits_per_image.detach(), dim=1)[1].cpu().numpy()
                text_label = labels.detach().cpu().numpy()
                text_predict = torch.max(logits_per_text.detach(), dim=1)[1].cpu().numpy()
                train_acc = (flat_accuracy(image_predict, image_label) + flat_accuracy(text_predict, text_label))/2.0
                #true = torch.cat([labels.detach(), labels.detach()])
                #predict = torch.cat([torch.max(logits_per_image.detach(), dim=1)[1], torch.max(logits_per_text.detach(), dim=1)[1]])
                #print ("true=", true)
                #print ("predict=", predict)
                #train_acc = metrics.accuracy_score(true, predic)
                #train_acc = torch.sum(true == predict)/true.shape[0]
                print ("train_acc=", train_acc)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # model恢复为训练模式
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
#test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    #print("Precision, Recall and F1-Score...")
    #print(test_report)
    #print("Confusion Matrix...")
    #print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    correct_sum = 0
    total_sum = 0
    total_eval_accuracy = 0
    with torch.no_grad():
        for images, text in data_iter:
            logits_per_image, logits_per_text = model(images.to(config.device), text.to(config.device))
            labels = torch.arange(logits_per_image.shape[0], device = config.device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            loss_total += loss
            
            
            image_label = labels.detach().cpu().numpy()
            image_predict = torch.max(logits_per_image.detach(), dim=1)[1].cpu().numpy()
            text_label = labels.detach().cpu().numpy()
            text_predict = torch.max(logits_per_text.detach(), dim=1)[1].cpu().numpy()
            total_eval_accuracy += (flat_accuracy(image_predict, image_label) + flat_accuracy(text_predict, text_label))/2.0
            
    
    avg_val_accuracy = total_eval_accuracy / len(data_iter)
    avg_val_loss = loss_total / len(data_iter)
    #print (labels_all)
    #print (correct_sum)
    return avg_val_accuracy, avg_val_loss

# 如果数据集是交付于华为云的validates数据集，则运行这个load函数
def load_data(parent_dir_name, config):    
    def load_zh_en_concepts(infilename):
        concept_dict = {}
        with open(infilename, "r", encoding = "utf-8") as infile:
            for line in infile:
                line_list = line.strip("\n").split("\t")
                if len(line_list) >= 2:
                    concept_dict[line_list[0]] = line_list[1]
        return concept_dict

    def load_concept_image_pair(infilename, select_concept_list, concepts_dict):
        train_image_conept_pair = {}
        validate_image_concept_pair = {}
        with open(infilename, "r", encoding = "utf-8") as infile:
            line = infile.readline()
            concept_dict = json.loads(line.strip())

        for key, value_list in concept_dict.items():
            for value in value_list:
                image_fielpath_jpg = parent_dir_name + "/images/" + str(value) + ".jpg"
                image_fielpath_png = parent_dir_name + "/images/" + str(value) + ".png"
                if os.path.isfile(image_fielpath_jpg):
                    image_filepath = image_fielpath_jpg
                elif os.path.isfile(image_fielpath_png):
                    image_filepath = image_fielpath_png
                else:
                    print ("path not exist, path=", image_fielpath_jpg, image_fielpath_png)
                    continue
                    
                if key in select_concept_list:
                    if image_filepath not in train_image_conept_pair and key in concepts_dict:
                        train_image_conept_pair[image_filepath] = concepts_dict[key]
                else:
                    if image_filepath not in validate_image_concept_pair and key in concepts_dict:
                        validate_image_concept_pair[image_filepath] = concepts_dict[key]
        # print ("train_image_conept_pair=", train_image_conept_pair)
        # print ("validate_image_concept_pair=", validate_image_concept_pair)
        return train_image_conept_pair, validate_image_concept_pair
    
    
    train_labels_list = []
    train_data_list = []
    validate_labels_list = []
    validate_data_list = []
    
    concepts_dict = load_zh_en_concepts("evaluation/validates/zh_en_concepts.txt")
    concepts_list = [key for key in concepts_dict.keys()]
    random.shuffle(concepts_list)
    train_concepts_len = int(len(concepts_list)*0.2)
    train_concepts_list = concepts_list[:train_concepts_len]
    train_image_conept_pair, validate_image_concept_pair = load_concept_image_pair("evaluation/validates/concept_image.json", train_concepts_list, concepts_dict)
    

    for key, value in train_image_conept_pair.items():
        train_data_list.append([key, value])
        train_labels_list.append(value)
    for key, value in validate_image_concept_pair.items():
        validate_data_list.append([key, value])
        validate_labels_list.append(value)

    # 保存finetune训练集
    with open("evaluation/train_concepts.txt", "w", encoding = "utf-8") as outfile:
        outfile.write("\n".join(train_labels_list))
    
    # 保存zero-shot的验证集
    with open("evaluation/validate_concepts.txt", "w", encoding = "utf-8") as outfile:
        outfile.write("\n".join(validate_labels_list))

    return train_data_list, len(set(train_labels_list)), validate_data_list, len(set(validate_labels_list))
    
            
        
# 如果是运行google搜索得到的图片对，则运行这个函数
def load_data_general(parent_dir_name, config):
    images_list = []
    labels_list = []
    data_list = []
    for dirname, _, filenames in os.walk(parent_dir_name):
        for file in filenames:
            file_path = os.path.join(dirname, file)
            label_ele_list = dirname.split("/")[-1].split(" ")
            if label_ele_list[-1] == "cartoon":
                original_label = "_".join(label_ele_list[:-1])
            else:
                original_label = "_".join(label_ele_list)

            ext = file.split(".")[-1]

            if ext == "lnk":
                continue

            #img = preprocess(Image.open(file_path))   # shape=[3,224,224]
            #label = clip.tokenize(original_label) # shape=[1,77]

            data_list.append([file_path, original_label])
            labels_list.append(original_label)

    return data_list, len(set(labels_list))

def build_dataset(data_list, ratios =[0.8,0.1,0.1]):
    length = len(data_list)
    train_size, validate_size = int(ratios[0]*length), int(ratios[1]*length)
    test_size = length - train_size - validate_size

    random.shuffle(data_list)
    train_data, validate_data, test_data = data_list[:train_size], data_list[train_size: train_size + validate_size], data_list[train_size + validate_size:]

    return train_data, validate_data, test_data
    

def transfer_unduplicate_sample(batches, batch_size):
    # print (type(batches))  # <class 'list'>
    # print (type(batches[0])) # <class 'list'>
    # print (type(batches[0][0])) # <class 'torch.Tensor'>
    # print (len(batches))   # 
    # print (len(batches[0]))
    # tranfer batches --> new batches
    # requirement: 1.每一个batch内的image和text都必须完全不同。
    #              2.所有batch，一个image和text出现的次数不超过floor(classes/batch_size)
    total_count = len(batches)
    n_batches = total_count // batch_size + 1   # //: floor除法(向下取整除),返回除数  7//2=3

    batches_by_classes_dict = {}

    tranfer_batches = []
    is_visited = np.zeros(total_count)

    max_list_len = 0
    for item_index in range(len(batches)): 
        _, text = batches[item_index]
        if text not in batches_by_classes_dict:
            batches_by_classes_dict[text] = [item_index]
        else:
            batches_by_classes_dict[text].append(item_index)
        
        if max_list_len < len(batches_by_classes_dict[text]):
            max_list_len = len(batches_by_classes_dict[text])
    
    classes_num = len(batches_by_classes_dict.keys())
    if batch_size > classes_num:
        print ("batch_size={0} is larger than class_num={1}. Let batch_size be equal to class_num, "
        "that is batch_size={1}".format(batch_size, classes_num))
        batch_size = classes_num
    
    # 根据max_list_len，补全不足长度的list
    for text, item_index_list in batches_by_classes_dict.items():
        item_index_list_len = len(item_index_list)
        if item_index_list_len < max_list_len:
            diff_num = max_list_len - item_index_list_len
            diff_times = diff_num // item_index_list_len
            for time_num in range(diff_times):
                item_index_list.extend(random.sample(item_index_list, item_index_list_len))
            if diff_num % item_index_list_len != 0:
                item_index_list.extend(random.sample(item_index_list, diff_num % item_index_list_len))


    for list_index in range(max_list_len):
        candidate_batch_items = [batches[item_index_list[list_index]] for item_index_list in batches_by_classes_dict.values()]
        #print ("candidate=", candidate_batch_items)
        for item_index in range(len(candidate_batch_items)):
            copy_candidate_batch_items = candidate_batch_items.copy()
            candidate_item = copy_candidate_batch_items.pop(item_index)
            # print (len(candidate_item))
            # print (type(candidate_item))
            # print (type(candidate_item[0]))
            
            per_transfer_batch = []
            per_transfer_batch.append(candidate_item)
            # batch_size < class_num
            per_transfer_batch.extend(random.sample(copy_candidate_batch_items, batch_size - 1))
            #print ("len of per_transfer_batch=", len(per_transfer_batch))
            #tranfer_batches.append(per_transfer_batch)
            tranfer_batches.extend(per_transfer_batch)

    # print (type(tranfer_batches))  # <class 'list'>
    # print (type(tranfer_batches[0])) # <class 'list'>
    # print (type(tranfer_batches[0][0])) # <class 'torch.Tensor'>
    return tranfer_batches
    

class DatasetIterater(object):
    def __init__(self, batches, config):
        self.batch_size = config.batch_size
        #self.origional_batches = batches
        #self.batches = transfer_unduplicate_sample(batches, config.batch_size)
        self.batches = batches
        self.n_batches = len(batches) // config.batch_size    # //: floor除法(向下取整除),返回除数  7//2=3
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # %: 取模运算，返回除法的余数
            self.residue = True
        self.index = 0
        self.device = config.device

    def _to_tensor(self, datas):
        images_list = []
        text_list = []
        #print ("len=", len(datas))
        for file_path, label in datas:
            # print (file_path, label)
            if file_path.split(".")[-1] == "jpg":
                image_code = Image.open(file_path)
            else:
                image_code = Image.open(file_path).convert("RGB")
            #img = preprocess(Image.open(file_path))#.tolist()                 # shape=[3,224,224]
            img = preprocess(image_code)#.tolist()                 # shape=[3,224,224]
            text = clip.tokenize("a photo of a " + label)#.tolist()           # shape=[1,77]
            images_list.append(img)
            text_list.append(text[0])
        #print (len(images_list))
        #print (len(text_list))
        images_input = torch.stack(images_list, dim=0).to(self.device)
        text_input = torch.stack(text_list, dim=0).to(self.device)
        
        return (images_input, text_input)
        #return datas

    def __next__(self):
        if self.residue and self.index == self.n_batches: # 最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            # print (type(self.batches))
            # print (len(self.batches))
            # print (type(self.batches[0]))
            # print (len(self.batches[0]))
            if _DEBUG:
                print ("DatasetIterater per image=", self.batches[0][0].shape)
                print ("DatasetIterater per text=", self.batches[0][1].shape)
            # print (self.index * self.batch_size)
            # print ((self.index + 1) * self.batch_size)
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 众多参数
config = Config()
# initiate model 
model, preprocess = clip.load("ViT-B/32", device=config.device, jit=False)  # 确定训练设备
#model, preprocess = clip.load("/home/featurize/work/CLIP/save/contrastive_learning_checkpoints/CLIP-concept-finetune_clustering_modified_classified_0_8.ckpt", device=config.device, jit=False)  # 确定训练设备

# data process
start_time = time.time()
print("Loading data...")

# 获得数据集 
# 数据格式1： dir/dir_label/file_label_index.ext
# parent_dir_name = "data/img_all_small_bak_fewer/"
#data_list, classes_num = load_data_general(parent_dir_name, config)  # [[image_tensor, label_tensor], ....]
# 数据格式2: dir/index.ext   (concept_image_pair.json)
parent_dir_name = "evaluation/validates/"
data_list, classes_num, validate_data_list, validate_class_num = load_data(parent_dir_name, config)  # [[image_tensor, label_tensor], ....]
#print ("train_data_list=", data_list)
#print ("train_class_num=", classes_num)
#print ("validate_data_list=", validate_data_list)
#print ("validate_class_num=", validate_class_num)

config.classes_num = classes_num

data_list = transfer_unduplicate_sample(data_list, config.batch_size)

# 切分训练集和验证集
train_data, dev_data, test_data = build_dataset(data_list, ratios=[0.85,0.14,0.01])  # 数据集预处理
print (len(train_data))
print (len(dev_data))
print (len(test_data))

# 根据batch切分好tensor
train_iter = build_iterator(train_data, config)    
dev_iter = build_iterator(dev_data, config)    
test_iter = build_iterator(test_data, config)    

time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
train(config, model, train_iter, dev_iter, test_iter)  # 开始训练
#print (len(train_data))
#train_data = np.array(train_data)
#train(config, model, train_data)
print ("finish training")

print ("start validation")
_, _, validate_test_data = build_dataset(data_list, ratios=[0.01,0.01,0.98])
validate_test_iter = build_iterator(validate_test_data, config)
test(config, model, validate_test_iter)
