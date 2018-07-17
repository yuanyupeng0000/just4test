import os
import random

trainval_percent = 0.8    #  修改训练集与测试集比例，此时train:test=8:2
train_percent = 0.7       #  train 占 trainval 中的 0.7 ， 后面只用 trainval，所以这里这个数值不重要
fdir = '/home/yyp/data/VOCdevkit/VOC2007/ImageSets/Main/'      # 修改对应路径
xmlfilepath = '/home/yyp/data/VOCdevkit/VOC2007/Annotations/'  # 修改对应路径
txtsavepath = fdir
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open(fdir + 'trainval.txt', 'w')
ftest = open(fdir + 'test.txt', 'w')
ftrain = open(fdir + 'train.txt', 'w')
fval = open(fdir + 'val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
