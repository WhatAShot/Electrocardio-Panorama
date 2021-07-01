天池数据介绍页面：https://tianchi.aliyun.com/competition/entrance/231754/information

PTB数据介绍页面：

interval格式说明：

interval文件为一个个json文件，文件名跟心电数据集的原始数据命名一致，只是后缀的不同

interval对于一个beat的心电数据标记6个点，分别为P波的起点终点，QRS波(还是R波？以文章描述为准)的起点终点，T波的起点终点。这6个点分别对应json文件中的key：["P on", "P off", "R on", "R off", "T on", "T off"]。每个key对应的value是一个list，里面是按照时间顺序记录的对应波形特征点的所在时刻。一个心拍中的标注点的组合就构成了相应的interval。

标注interval尽量保证所在波形的完整性，即对于一份case，可能存在最前面跟最后面的心拍是记录不全的，对于这种心拍，不做标注。但是可能标注数据中还是有少量的interval是标注了这种波形，并且很有可能这个心拍是标注不全的，比如缺少P波的起点终点。这可以通过编写相应的数据清洗代码来解决。



在google drive中下载我们最优的权重(
https://drive.google.com/file/d/1S6gNrIjtFH0WGjgsmEHNr4OgtDy9L3dS/view?usp=sharing

)，并且放置于codes/output/weight/nef_net/nef_net下

即

cp -r {your_folder}/best_valid.pkl codes/output/weight/nef_net/nef_net

运行demo.ipynb即可



在
https://drive.google.com/file/d/1tMTY-6LOxt1gSIn4jCi1BDO3EfL6CeOe/view?usp=sharing
下载ptb数据，包含原始数据经过预处理之后的每一个心拍的数据及其interval数据，解压之后放在

data/tianchi/npy_data/pkl_data/train_heartbeats.pkl 以及 data/tianchi/npy_data/pkl_data/test_heartbeats.pkl 即可使用ptb数据集

