from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import preprocessing_images
import preprocessing_captions

'''
按照模型要求，生成相应的输入、输出批次
输入：x=[x1,x2]
    x1:(2048,)图片特征向量
    x2:(34,)图片对应的标题一部分（从startseq开始，一次增加一个词）
输出：
    y:(34),x2中下一个单词的tokenid
'''

def data_generator(image_fea,descriptions,image_idlist,word2ix,max_cap_length = 34,vocab_len = 1949,photo_num_per_batch = 2,shuffle = False):
    x1,x2,y = list(),list(),list()

    while(True):
        num_img = 0
        if shuffle:
            np.random.shuffle(image_idlist)

        for image_id in image_idlist:
            num_img += 1
            x1_fea = image_fea[image_id]
            desc_list = descriptions[image_id]
            for desc in desc_list:
                desc = [ w for w in desc.split() if w in word2ix.keys()]
                for i in range(1,len(desc)):
                    x1.append(x1_fea)
                    partial_cap = [ word2ix[w] for w in desc[:i] ]
                    x2.append(partial_cap)
                    y.append([word2ix[desc[i]]])

            if num_img == photo_num_per_batch:
                x2 = pad_sequences(np.array(x2), maxlen=max_cap_length, padding='post',value = 592)
                y = to_categorical(np.array(y), num_classes=vocab_len)
                x1 = np.array(x1)
                yield ([x1,x2],y)
                num_img = 0
                x1, x2, y = list(), list(), list()

def load_image_id(image_testfile):
    image_idset = list()
    for line in preprocessing_captions.doc_generator(image_testfile):
        image_idset.append(line.split('.')[0])
    return image_idset

if __name__ == '__main__':
    description_file = 'dataset/Flickr8k_text/Flickr8k.token.txt'
    dectionary_file = 'dataset/Pickle/simple_dictionary.pkl'

    descriptions = preprocessing_captions.load_descriptions(description_file)
    descriptions = preprocessing_captions.clean_descriptions(descriptions)

    word2ix, ix2word = preprocessing_captions.loac_dictionary(dectionary_file)


    train_image_fea_file = 'dataset/Pickle/encoded_train_images.pkl'
    train_textfile = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'

    test_image_fea_file = 'dataset/Pickle/encoded_test_images.pkl'
    test_textfile = 'dataset/Flickr8k_text/Flickr_8k.testImages.txt'

    train_img_idlist = load_image_id(train_textfile)
    train_img_fea = preprocessing_images.load_image_fea(train_image_fea_file)

    n = 0
    for ([x1,x2],y) in data_generator(image_fea=train_img_fea,descriptions = descriptions,image_idlist=train_img_idlist,
                             word2ix=word2ix,max_cap_length=34,vocab_len=1949,photo_num_per_batch=2,shuffle=True):
        # print(x1.shape)
        print(' '.join( [ix2word[ix] for ix in x2[5] if ix in ix2word] ))
        print( ix2word[int(np.argmax(y[0]))]  )

        n += 1
        if n == 3:
            break

    print('ok')