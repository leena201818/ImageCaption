本文原版参考Harshall Lamba的作品实现
博客地址：https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
代码地址：https://github.com/hlamba28/Automatic-Image-Captioning

打印keras模型，需要安装graphviz：apt-get install graphviz；需要安装pip install pydot;
本文用到glove词嵌入模型参数：https://nlp.stanford.edu/projects/glove/

to do:
1、增加Flicker30k进行测试
2、Image嵌入层和caption嵌入层采用拼接方式
3、词汇表词频，如何处理pad_sequence中填充问题
4、对比im2txt方法