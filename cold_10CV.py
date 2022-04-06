import pandas as pd
import numpy as np
import pickle
import os
import random
from random import randint

#1、导入数据
def load_Smiles():
    Datas = 'D:\模型们\SSI2_1\CID_smiles.txt'
    Smiles = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        lineArr = line.strip().split()
        Smiles.append(lineArr[1])
        line = fileIn.readline()
    return Smiles

def neg():
    neg = []
    neg1 = randint(0,554)
    neg2 = randint(0,554)
    inte = randint(0,1317)
    lab = 0
    neg.append(neg1)
    neg.append(neg2)
    neg.append(inte)
    neg.append(lab)
    return neg

def load_DDIs():
    Datas = 'D:\模型们\SSI2_1\DDI.txt'
    DDIs = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        lineArr = line.strip().split()
        lineArr.append(1)
        DDIs.append(list(map(int,lineArr)))
        DDIs.append(neg())
        line = fileIn.readline()
    return DDIs

#2、分割药物列表，生成10倍交叉验证文件
def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list

def split_list(alist, group_num=4, shuffle=True, retain_left=True):
    index = list(range(len(alist)))  # 保留下标
    # 是否打乱列表
    if shuffle:
        random.shuffle(index)
    elem_num = len(alist) // group_num  # 每一个子列表所含有的元素数量
    sub_lists = {}
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists['set' + str(idx)] = subset(alist, index[start:end])
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index):  # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set' + str(idx + 1)] = subset(alist, index[end:])
    return sub_lists

def drug_sets():
    drug_index = []
    for i in range(0,555):
        drug_index.append(i)
    drug_set = split_list(drug_index, group_num=10)
    for i in drug_set['set10']:
        drug_set['set1'].append(i)
    return drug_set

drug_set1 = [308, 165, 22, 167, 34, 273, 196, 490, 274, 252, 506, 319, 120, 247, 322, 54, 528, 76, 187, 382, 191, 514, 13, 122, 352, 367, 52, 541, 14, 280, 249, 50, 239, 475, 470, 67, 115, 11, 290, 105, 117, 544, 330, 439, 30, 345, 522, 162, 157, 315, 442, 71, 534, 364, 417]
drug_set2 = [42, 493, 386, 31, 309, 326, 83, 410, 251, 478, 357, 44, 219, 361, 130, 400, 5, 129, 26, 65, 104, 503, 152, 338, 236, 143, 421, 284, 314, 299, 377, 351, 135, 501, 450, 77, 282, 128, 520, 383, 349, 370, 198, 304, 526, 483, 40, 540, 131, 508, 73, 16, 445, 449, 192, 464, 231, 155, 125, 446]
drug_set3 = [515, 408, 481, 45, 194, 409, 437, 158, 243, 295, 354, 25, 334, 300, 99, 343, 547, 549, 270, 43, 109, 180, 302, 296, 230, 429, 107, 102, 91, 342, 35, 498, 454, 305, 536, 441, 455, 48, 156, 151, 289, 462, 250, 216, 217, 436, 434, 425, 509, 380, 37, 545, 339, 106, 244]
drug_set4 = [121, 233, 440, 195, 41, 49, 58, 92, 265, 329, 505, 535, 395, 513, 132, 166, 531, 278, 24, 539, 297, 242, 150, 36, 346, 461, 324, 479, 19, 56, 373, 495, 482, 546, 159, 404, 87, 396, 537, 148, 344, 538, 39, 21, 397, 96, 15, 3, 317, 476, 6, 84, 138, 221, 285]
drug_set5 = [97, 95, 353, 363, 507, 413, 460, 197, 163, 110, 82, 268, 142, 283, 516, 389, 70, 18, 245, 471, 372, 100, 61, 402, 269, 281, 264, 473, 321, 214, 38, 418, 257, 224, 223, 47, 333, 93, 291, 46, 137, 203, 433, 208, 27, 234, 80, 356, 403, 359, 171, 423, 182, 335, 459]
drug_set6 = [376, 59, 401, 488, 237, 524, 341, 543, 452, 477, 175, 114, 474, 331, 384, 240, 312, 98, 153, 337, 190, 318, 320, 511, 487, 146, 484, 323, 174, 369, 463, 415, 529, 496, 287, 72, 20, 392, 260, 502, 126, 199, 466, 387, 253, 430, 542, 124, 480, 275, 101, 424, 523, 306, 261]
drug_set7 = [210, 288, 136, 201, 340, 491, 154, 57, 179, 222, 316, 134, 161, 218, 453, 90, 368, 207, 271, 79, 183, 348, 393, 55, 127, 438, 238, 510, 258, 554, 69, 272, 186, 451, 168, 23, 78, 86, 405, 398, 68, 465, 225, 485, 375, 362, 89, 394, 388, 550, 365, 527, 170, 185, 189]
drug_set8 = [118, 366, 548, 391, 457, 411, 407, 518, 202, 525, 7, 133, 259, 431, 327, 311, 519, 81, 139, 358, 512, 164, 456, 9, 504, 350, 169, 426, 248, 332, 29, 108, 193, 444, 406, 355, 336, 94, 53, 379, 416, 301, 521, 286, 32, 205, 292, 458, 0, 310, 28, 419, 390, 226, 517]
drug_set9 = [144, 551, 307, 51, 88, 215, 500, 12, 4, 262, 145, 177, 200, 211, 75, 378, 328, 530, 381, 467, 229, 206, 213, 141, 63, 427, 293, 160, 111, 2, 492, 116, 422, 533, 254, 33, 176, 123, 74, 360, 472, 85, 374, 188, 212, 497, 1, 414, 494, 172, 385, 303, 298, 66, 209]
drug_set10 = [256, 294, 532, 64, 399, 553, 228, 267, 468, 347, 147, 325, 246, 279, 178, 486, 241, 448, 173, 113, 181, 8, 255, 103, 17, 184, 263, 432, 10, 149, 235, 204, 140, 469, 552, 443, 435, 62, 277, 499, 119, 60, 266, 313, 227, 276, 232, 420, 412, 112, 447, 371, 428, 220, 489]

def write_to_txt():
    DDs = load_DDIs()
    Smiles = load_Smiles()
    for DD in DDs:
        if DD[0] not in drug_set1 and DD[1] not in drug_set1:
            with open('cold_train.txt', 'a') as f1:
                f1.write(str(DD[0])+' '+str(DD[1])+' '+Smiles[DD[0]]+' '+Smiles[DD[1]]+' '+str(DD[2])+' '+str(DD[3])+'\n')
        if DD[0] in drug_set1 and DD[1] not in drug_set1:
            with open('cold_test_C2.txt', 'a') as f2:
                f2.write(str(DD[0]) + ' ' + str(DD[1]) + ' ' + Smiles[DD[0]] + ' ' + Smiles[DD[1]] + ' ' + str(DD[2]) + ' ' + str(DD[3]) + '\n')
        if DD[0] not in drug_set1 and DD[1] in drug_set1:
            with open('cold_test_C2.txt', 'a') as f2:
                f2.write(str(DD[0]) + ' ' + str(DD[1]) + ' ' + Smiles[DD[0]] + ' ' + Smiles[DD[1]] + ' ' + str(DD[2]) + ' ' + str(DD[3]) + '\n')
        if DD[0] in drug_set1 and DD[1] in drug_set1:
            with open('cold_test_C3.txt', 'a') as f3:
                f3.write(str(DD[0]) + ' ' + str(DD[1]) + ' ' + Smiles[DD[0]] + ' ' + Smiles[DD[1]] + ' ' + str(DD[2]) + ' ' + str(DD[3]) + '\n')
    return 0

#3、导入以上DDI数据，转换成可训练的pkl文件
def load_DDIs():
    Datas = 'D:\模型们\\topk_GNNSUB\cold_train.txt'
    DDI_index = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        id1, id2, smiles1, smiles2, interaction,label = line.strip().split()
        line = fileIn.readline()
        DDI_index.append(map(int,[id1, id2, interaction, label]))
    return DDI_index

def save_data(data, filename):
    dirname = f'D:/模型们/topk_GNNSUB'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')

def load_DD_data():
    with open(f'D:/模型们/topk_GNNSUB/drug_data.pkl', 'rb') as f:
        drugdata = pickle.load(f)
    Drug_pair = {}
    DDI_index = load_DDIs()
    for index,(drug1,drug2,interaction,label) in enumerate(DDI_index):
        x_1,edge_index_1,edge_feature_1 = drugdata[drug1]
        Drug1 = dict(x=x_1, edge_index = edge_index_1, edge_feature = edge_feature_1)
        x_2, edge_index_2, edge_feature_2 = drugdata[drug2]
        Drug2 = dict(x=x_2, edge_index = edge_index_2, edge_feature = edge_feature_2)
        drugpair = dict(drug_1 = Drug1, drug_2 = Drug2, Inter = interaction, Label = label)
        Drug_pair[index] = drugpair
    save_data(Drug_pair, 'cold_train_Pair.pkl')
    return Drug_pair




