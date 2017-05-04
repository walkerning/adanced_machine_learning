本次使用的数据文件包含author.txt，paper.txt，citation_train.txt 和 test.txt。

1、author.txt:  作者ID信息

     本文件包含作者ID和作者名字的一一映射关系（作者名字唯一确定），每一行由id和作者名组成，以“\t”隔开，例如：
     14210       Jiawei Han             

 

2、paper.txt:  论文数据

     该文件包含了2011年及之前所有论文的信息，不同论文以空行隔开，包含论文间引用关系但不完整。每篇论文具体字段格式如下（有些论文某些字段缺失）：

     #*            论文标题
     #@          作者，多作者以”, ”隔开
     #t            论文发表年份
     #c            论文发表的会议、期刊
     #index     论文的ID
     #%           该论文所引用的论文（包含多行，每行代表一篇被引用论文的ID）
     #!             论文摘要

     样例：
     #*Information geometry of U-Boost and Bregman divergence
     #@Noboru Murata, Takashi Takenouchi, Takafumi Kanamori,Shinto Eguchi
     #t2004
     #cNeural Computation
     #index436405
     #%94584
     #%282290
     #%605546
     #%620759
     #%564877
     #%564235
     #%594837
     #%479177
     #%586607
     #!We aim at an extension of AdaBoost to U-Boost, in the paradigm to build a stronger classification machine from a set of weak learning machines. A geometric understanding of the Bregman divergence defined by a generic convex function U leads to the U-Boost method in the framework of information geometry extended to the space of the finite measures over a label set. We propose two versions of U-Boost learning algorithms by taking account of whether the domain is restricted to the space of probability functions. In the sequential step, we observe that the two adjacent and the initial classifiers are associated with a right triangle in the scale via the Bregman divergence, called the Pythagorean relation. This leads to a mild convergence property of the U-Boost algorithm as seen in the expectation-maximization algorithm. Statistical discussions for consistency and robustness elucidate the properties of the U-Boost methods based on a stochastic assumption for training data.

   

3、citation_train.txt:  部分作者的总引用数

     本文件包含部分作者（约1,000,000）截止到2016年的引用数。数据的每一行由作者ID，作者名字和引用数组成，由”\t”隔开，例如：
     14210      Jiawei Han         126147

   

4、test_id.txt:  待预测的作者

     本文件为测试集，包含引用数未知待预测的作者, 作者名字和ID映射关系由author.txt给出。数据的每一行为作者ID及名字，例如：
     8    Krunoslav Puljic