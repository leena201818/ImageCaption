from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
# from PIL import Image as image
import os
import pickle
from time import time
import numpy as np
'''
    采用keras预训练模型，将图片转换成2048维特征向量，并保存
'''
model_weights_path = 'dataset/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'

image_base_path = 'dataset/Flicker8k_Dataset'
train_textfile = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
test_textfile =  'dataset/Flickr8k_text/Flickr_8k.testImages.txt'
val_textfile =  'dataset/Flickr8k_text/Flickr_8k.devImages.txt'

def model_extract_fea():
    model = InceptionV3(weights=model_weights_path, include_top=True)
    model_new = Model(inputs=model.input, outputs=model.layers[-2].output)
    return model_new

#用来提取图片特征的模型
g_model_extract_fea = model_extract_fea()

# 读取一张图片，按照InceptionV3要求进行预处理，返回参数（1,299,299,3）
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    # 将PIL格式的图片转换成3-d numpy array,根据backend不同，生成相应的(height,with,depth)
    x = image.img_to_array(img)
    # 增加batch_size维度:(1,height,with,depth)
    x = np.expand_dims(x, axis=0)
    # 进行InceptionV3预处理
    x = preprocess_input(x)

    return x

#提取一张图片的特征向量，返回参数(2048,)
def encode(image_path):
    img = preprocess(image_path)        #(1,299,299,3)
    feature_vec = g_model_extract_fea.predict(img)  #(1,2048)
    feature_vec = np.reshape(feature_vec,feature_vec.shape[1])

    return feature_vec

#从训练、测试txt中提取图片的id，装入set，避免重复
def load_image_set(filename):
    with open(filename,'r') as f:
        doc = f.read()
        image_idlist = [ line.split('.')[0] for line in doc.split('\n') if len(line) > 0]
        return set(image_idlist)

#将训练、测试图片的向量保存到文件，字典格式（imageid,feature_vec）
def dump_image_fea(image_path,imageid_set,pickle_file):
    start = time()
    encoding_image = {}
    global image_base_path
    for img in imageid_set:
        image_path = os.path.join(image_base_path,img+'.jpg')
        print('Extracting feature of {}'.format(image_path))
        encoding_image[img] = encode(image_path)

    print('Time taken in seconds = {}'.format(time() - start))

    with open(pickle_file,'wb') as encoded_pickle:
        pickle.dump(encoding_image,encoded_pickle)
    print('Dump the image feature vector to {}'.format(pickle_file))

#提取训练、测试图片的特征向量
def load_image_fea(pickle_file = 'dataset/Pickle/encoded_train_images.pkl'):
    with open(pickle_file,'rb') as encoded_pickle:
        return pickle.load(encoded_pickle)

if __name__ == '__main__':

    # imagep = 'dataset/Flicker8k_Dataset/3086810882_94036f4475.jpg'
    # a = preprocess(image_path=imagep)

    # 将图片转换成特征向量，并保存到文件
    # train_imgid_set = load_image_set(train_textfile)
    # print('convert training image:{}'.format(len(train_imgid_set)))
    # dump_image_fea(image_path=image_base_path,imageid_set=train_imgid_set,pickle_file='dataset/Pickle/encoded_train_images.pkl')
    #
    # test_imgid_set = load_image_set(test_textfile)
    # print('convert test image:{}'.format(len(test_imgid_set)))
    # dump_image_fea(image_path=image_base_path,imageid_set=test_imgid_set,pickle_file='dataset/Pickle/encoded_test_images.pkl')

    # val_imgid_set = load_image_set(val_textfile)
    # print('convert valadition image:{}'.format(len(val_imgid_set)))
    # dump_image_fea(image_path=image_base_path,imageid_set=val_imgid_set,pickle_file='dataset/Pickle/encoded_val_images.pkl')


    print('ok')
