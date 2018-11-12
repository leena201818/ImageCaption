from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt

import preprocessing_captions
import preprocessing_images
import data_generator
import caption_model

import numpy as np

description_file = 'dataset/Flickr8k_text/Flickr8k.token.txt'
dectionary_file = 'dataset/Pickle/simple_dictionary.pkl'

descriptions = preprocessing_captions.load_descriptions(description_file)
descriptions = preprocessing_captions.clean_descriptions(descriptions)

word2ix, ix2word = preprocessing_captions.loac_dictionary(dectionary_file)

train_image_fea_file = 'dataset/Pickle/encoded_train_images.pkl'
train_textfile = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'

test_image_fea_file = 'dataset/Pickle/encoded_test_images.pkl'
test_textfile = 'dataset/Flickr8k_text/Flickr_8k.testImages.txt'

val_image_fea_file = 'dataset/Pickle/encoded_val_images.pkl'
val_textfile =  'dataset/Flickr8k_text/Flickr_8k.devImages.txt'

train_img_idlist = data_generator.load_image_id(train_textfile)
train_img_fea = preprocessing_images.load_image_fea(train_image_fea_file)

test_img_idlist = data_generator.load_image_id(test_textfile)
test_img_fea = preprocessing_images.load_image_fea(test_image_fea_file)

val_img_idlist = data_generator.load_image_id(val_textfile)
val_img_fea = preprocessing_images.load_image_fea(val_image_fea_file)

dict_file = 'dataset/Pickle/simple_dictionary.pkl'
word2ix2,_ = preprocessing_captions.loac_dictionary(dict_file)

# 训练曲线
def show_history(history):
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history['acc'], label='train_acc')
    plt.plot(history.epoch, history.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()

def train(weight_file = 'dataset/model.h5'):

    model = caption_model.create_model(input_shape_img=(2048,), vocab_size=1949, caption_max_length=34, embedding_dim = 200)

    def scheduler(epoch):
        if int(epoch % 30) == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {} at epoch{}".format(lr * 0.5, epoch))
            return K.get_value(model.optimizer.lr)
        else:
            print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
            return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    batch_size = 3
    generator_train = data_generator.data_generator(image_fea=train_img_fea,descriptions=descriptions,
                                             image_idlist=train_img_idlist,word2ix=word2ix,max_cap_length=34,
                                             vocab_len=1949,photo_num_per_batch=batch_size,shuffle=True)

    generator_val = data_generator.data_generator(image_fea=val_img_fea, descriptions=descriptions,
                                                    image_idlist=val_img_idlist, word2ix=word2ix, max_cap_length=34,
                                                    vocab_len=1949, photo_num_per_batch=batch_size,shuffle=False)

    epoch = 20
    '''steps_per_epoch的含义：完成一个完整的epoch，需要进行steps_per_epoch次generator_train调用，一般为样本数/batch_size'''
    steps_per_epoch = int( len(train_img_idlist) / batch_size )

    history = model.fit_generator(generator_train, steps_per_epoch=steps_per_epoch,
                        epochs=epoch,
                        verbose=1,
                        callbacks=[reduce_lr,ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)],
                        validation_data=generator_val,
                        validation_steps = int( len(val_img_idlist) / batch_size ),
                        shuffle=False)

    show_history(history)

if __name__ == '__main__':
    train()