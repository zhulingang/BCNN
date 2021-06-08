# BCNN

BCNN菊花细粒度图像识别模型

## Requirements

- Python==3.7
- torch==1.7.0

## 训练步骤

1. 下载VGG16预训练模型和数据集

   若已下载，可省略此步骤

   [VGG16-Download](https://pan.baidu.com/s/1OkIuKosTRfcZlDXkOW4WLQ)

   下载菊花数据集

2. 修改读取数据集和VGG-16模型路径

   - 在train_last.py和train_finetune.py中修改VGG-16路径

   ![image-20210514155123685](https://raw.githubusercontent.com/zhulingang/BlogImage/main/images/20210514160111.png)

   - 在data.py中修改数据集路径

   ![image-20210514155247016](https://raw.githubusercontent.com/zhulingang/BlogImage/main/images/20210514160118.png)

   - 在train_last_old.py和train_finetune_pld.py修改读取图片txt的路径

   ![image-20210514155902746](https://raw.githubusercontent.com/zhulingang/BlogImage/main/images/20210514160123.png)

   - 在bilinear_model.py修改菊花的总共类数num_class

     ![image-20210514163037785](https://raw.githubusercontent.com/zhulingang/BlogImage/main/images/20210514163045.png)

     

     

3. 训练数据集

   1. python train_last_old.py

      运行train_last_old.py，得到bcnn_lastlayer.pth模型参数。old为GPU版

   2. python train_finetune_old.py

      运行train_finetune_old.py,得到bcnn_alllayer.pth模型参数

