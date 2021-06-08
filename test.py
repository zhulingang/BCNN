import os
# train_txt_path='/Users/zhulingang/Desktop/新菊花数据集/new_train.txt'
# fo1=open(train_txt_path,'w')
# path='/Users/zhulingang/Desktop/新菊花数据集/train'
# path_list=os.listdir(path)
# path_list.remove('.DS_Store')
# path_list.sort()
# def write_txt(dirname,path):
#     filelist = os.listdir(path)
#     if '.DS_Store' in filelist:
#         filelist.remove('.DS_Store')
#     for i in filelist:
#         fo1.write(dirname + '/' + i + '\n')
#
# for category in path_list:
#     # 判断是否是文件
#     inpath=os.path.join(path, category)
#     if os.path.isdir(inpath):
#         write_txt(category,inpath)
# fo1.close()



# def append_new(oldpath,newpath,model):
#     f1=open(oldpath,'r')
#     f2=open(newpath,'a+')
#     for line in f1:
#         line=model+'/'+line
#         f2.write(line)
#     f1.close()
#     f2.close()
#
# train_path='/Users/zhulingang/Desktop/新菊花数据集/train.txt'
# train_newpath='/Users/zhulingang/Desktop/新菊花数据集/train_1.txt'
# model='train'
# append_new(train_path,train_newpath,model)

fh = open('/Users/zhulingang/Desktop/新菊花数据集/new_train_1.txt', 'r')
for line in fh:
    line = line.strip('\n')
    line = line.rstrip()
    location=line.find('/')
    print(line)
    print(int(line[location+1:location+4])-1)
