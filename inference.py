from keras.models import load_model
import preprocessing_images
import preprocessing_captions
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import argparse

image_base_path = 'dataset/Flicker8k_Dataset'
test_textfile =  'dataset/Flickr8k_text/Flickr_8k.testImages.txt'

test_imgid_set = preprocessing_images.load_image_set(test_textfile)
test_img_fea = preprocessing_images.load_image_fea(pickle_file='dataset/Pickle/encoded_test_images.pkl')

dectionary_file = 'dataset/Pickle/simple_dictionary.pkl'
word2ix,ix2word = preprocessing_captions.loac_dictionary(dectionary_file)

weight_file = 'dataset/model.h5'
model = load_model(weight_file)


'''
    预测图片标题
    给定测试集合中的一个图片特征，然后预测标题
'''
def inference_from_feature(img_fea):
    start_id = word2ix['startseq']
    end_id = word2ix['endseq']

    caption_ids = [start_id]

    for i in range(34-1):
        caption_ids_seq = pad_sequences([caption_ids], maxlen=34, padding='post',value=592)[0]
        y = model.predict([[img_fea],[caption_ids_seq]],verbose=0)
        next_wordid = np.argmax(y)
        caption_ids.append( next_wordid )

        if next_wordid == end_id:
            break
    caption = [ix2word[id] for id in caption_ids ]
    return ' '.join(caption[1:-1])
'''
    预测图片标题
    给定测试集合中的一个图片ID,从已经预存的图片特征中提取特征，然后预测标题
'''
def inference_from_imageid(image_id):
    img_fea = test_img_fea[image_id]
    return inference_from_feature(img_fea)

'''
    预测图片标题
    给定一张任意图片，实时提取特征，预测标题
'''
def inference_file(image_file):
    assert os.path.exists(image_file)
    img_fea = preprocessing_images.encode(image_file)
    return inference_from_feature(img_fea)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str,
                        help='图片路径(*.jpg)')
    #
    # parser.add_argument('-acc', '--accuracy',
    #                     help='显示正确率曲线', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':

    if len( sys.argv ) >1:
        args = parse_arguments(sys.argv[1:])
        if not os.path.exists(args.filepath) :
            print('输入的文件路径不存在！')
            exit(0)
        caption = inference_file(args.filepath)
        print('The title of {} is:{}'.format(args.filepath,caption))

    else:
        # test_result_file = 'dataset/testImagesCaption.txt'
        #
        # for line in preprocessing_captions.doc_generator(test_result_file):
        #     photo = os.path.join(image_base_path,line.split()[0])
        #     caption = line.split()[1:]
        #     caption = ' '.join(caption)
        #     plt.imshow(load_img(photo))
        #     plt.title(caption)
        #     plt.show()
        #     # h = input('any key press')


        # '''
        #     测试集生成标题到指定文件
        # '''
        test_result = {}
        test_result_file = 'dataset/testImagesCaption.txt'



        #仅仅显示少量图片
        row,col = 4,2
        plt.figure(row*col,figsize=(12,8))
        n = 1
        for line in preprocessing_captions.doc_generator(test_textfile):
            photo_file = os.path.join(image_base_path,line)
            photo = load_img(photo_file)

            image_id = line.split('.')[0]
            caption = inference_from_imageid(image_id)

            test_result[line] = caption

            if n <= row*col:
                plt.subplot(row,col,n)
                plt.imshow(photo)
                plt.title(caption)
            # else:
            #     break
            n+=1

        plt.show()

        #保存结果到文件
        with open(test_result_file,'w') as f:
            for photo,caption in test_result.items():
                f.write('{}     {}\n'.format(photo,caption))

        print('OK')