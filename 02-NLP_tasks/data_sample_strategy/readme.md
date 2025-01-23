实现简易模型，验证数据动态采样策略

tips
- dataset构建需要自己实现`__len__`和`__getitem__`方法
- datacollator是将数据组织成batch的形式
- model需要自己实现`forward`和`save_prtrained`方法
- trainer需要自己实现`_get_train_sampler`不然默认继承的Trainer是随机采样，同样需要自己实现`compute_loss`方法


流程：
01-data_generate.ipynb

sh 02-run_bash.sh