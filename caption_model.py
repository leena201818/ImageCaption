from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers import Add
from keras.models import Model
from keras.utils import plot_model
import numpy as np

import preprocessing_captions

glove_file = 'dataset/glove.6B.200d.txt'

dectionary_file = 'dataset/Pickle/simple_dictionary.pkl'
word2ix, _ = preprocessing_captions.loac_dictionary(dectionary_file)

'''
    glove预训练模型参数
    返回类型：字典{vocab_word:200d-ndarray}
'''
def embedding_weight(glove_file = 'dataset/glove.6B.200d.txt'):
    embeddings_index = {}
    with open(glove_file,'r',encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coef = np.asarray( values[1:],dtype='float32' )
            embeddings_index[word] = coef

    return embeddings_index

'''
    词嵌入层权重
    返回(vocab_size,embedding_dim)的ndarray
'''
def vocab_embedding_matrix(word2ix,glove_file = 'dataset/glove.6B.200d.txt'):
    glove_embedding_weight = embedding_weight(glove_file)

    vocal_size = len(word2ix.keys())
    embedding_dim = glove_embedding_weight['the'].shape[0]
    vocab_embedding_matrix = np.zeros((vocal_size,embedding_dim))

    for word,ix in word2ix.items():
        word_vector = glove_embedding_weight.get(word,np.zeros(embedding_dim))
        vocab_embedding_matrix[ix] = word_vector
    return vocab_embedding_matrix

'''
    模型输入图片和部分标题，输出标题的下一个词
    图片特征向量采用InceptionV3的倒数第二层,(2048,)向量，部分标题采用(34,)的序列
    嵌入层采用200d Glove与训练过词向量
    参数：
        input_shape_img:图片特征向量形状
        vocab_size:词汇表大小
        max_length:标题序列的最大长度
'''
def create_model(input_shape_img=(2048,),vocab_size = 1652,caption_max_length = 34,embedding_dim = 200):
    #图片特征向量
    input1 = Input(shape=input_shape_img,name='input_image')
    x1 = Dropout(0.5)(input1)
    x1 = Dense(units=256,activation='relu',name='dense_image')(x1)

    #部分标题序列
    input2 = Input(shape=(caption_max_length,),name='input_caption')
    x2 = Embedding(input_dim=vocab_size,output_dim=embedding_dim,mask_zero=True,name='embedding_caption')(input2)
    x2 = Dropout(0.5)(x2)
    x2 = LSTM(units=256,name='lstm')(x2)

    #两部分输入相加
    x = Add()([x1,x2])
    x = Dense(units=256,activation='relu',name='dense')(x)
    output = Dense(units=vocab_size,activation='softmax',name='output')(x)

    model = Model(inputs=[input1,input2],outputs = output)

    embedding_weight = vocab_embedding_matrix(word2ix, glove_file='dataset/glove.6B.200d.txt')
    model.get_layer('embedding_caption').set_weights([embedding_weight])
    model.get_layer('embedding_caption').trainable = False

    return model

if __name__ == '__main__':



    model = create_model(input_shape_img = (2048,),vocab_size = 1949,caption_max_length = 34)
    model.summary()
    #将模型结构打印，需要安装pip install pydot;apt-get install graphviz
    plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)

    model.predict()






