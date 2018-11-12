import string
import pickle

#文本文件行生成器,一次返回一行文本
def doc_generator(filename):
    with open(filename,'r') as f:
        doc = f.read().split('\n')
        for line in doc:
            if len(line) > 0:
                yield line

#字典{图片-描述列表}
def load_descriptions(filename):
    mapping = {}
    for line in doc_generator(filename):
        tokens = line.split()
        if len(tokens) > 1:
            image_id,image_desc = tokens[0],' '.join(tokens[1:])
            image_id = image_id.split('.')[0]
            if image_id not in mapping:
                mapping[image_id] = list()

            mapping[image_id].append(image_desc)
    return mapping
#小写、删除停用词、数字、单字母，返回清洗后的{图片-描述列表}
def clean_descriptions(descriptions):
    #字符转换表，三个参数，表明第三个参数中的字符会被转换成None，该表用在strval.translate(table)
    table = str.maketrans('','',string.punctuation)
    for image_id,image_des_list in descriptions.items():
        for i in range(len(image_des_list)):
            desc = image_des_list[i]
            desc = desc.split()
            desc = [w.lower() for w in desc]
            desc = [w.translate(table) for w in desc]
            desc = [w for w in desc if len(w) > 1]
            desc = [w for w in desc if w.isalpha()]
            #人为添加标题的开始和结束标志
            image_des_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'
    return descriptions

#标题最大长度
def max_length(descriptions):
    all_desc = list()
    for image_id,desc_list in descriptions.items():
        [all_desc.append(desc) for desc in desc_list]

    return max( [ len(desc.split()) for desc in all_desc] )


#词汇表，从全部描述中
def to_vocabulary(descriptions):
    all_vocab = set()
    for image_id in descriptions.keys():
        [all_vocab.update( desc.split() ) for desc in descriptions[image_id]]
    return all_vocab

#精简词汇表，从全部描述中提出词频低于threadhold的词
def cut_vocabulary(descriptions,threadhold_count = 10):
    wordcount = {}
    for image_id in descriptions.keys():
        desc_list = descriptions[image_id]
        for desc in desc_list:
            for word in desc.split():
                wordcount[word] = wordcount.get(word, 0) + 1

    all_vocab_list = list( to_vocabulary(descriptions) )
    cut_vocab = [word for word in all_vocab_list if wordcount[word] >= threadhold_count]

    return cut_vocab

#字典映射，并保存如文件，固化
def vocab_mapping(cut_vocab,save_filename):
    word2ix = {}
    ix2word = {}
    for i,word in enumerate(cut_vocab,start = 0):
        word2ix[word] = i
        ix2word[i] = word
    with open(save_filename,'wb') as f:
        pickle.dump([word2ix,ix2word],f)
    return word2ix,ix2word

#从文件中加载字典
def loac_dictionary(filename):
    with open(filename,'rb') as f:
        [word2ix,ix2word] = pickle.load(f)
    return [word2ix,ix2word]

if __name__ == '__main__':
    filename = 'dataset/Flickr8k_text/Flickr8k.token.txt'
    for i,l in enumerate( doc_generator(filename) ):
        print(l)
        if i > 1:
            break
    mapping = load_descriptions(filename)

    print('原始图片数，一图对应5个标题: %d ' % len(mapping))
    print( mapping['997722733_0cb5439472'] )

    mapping = clean_descriptions(mapping)
    print('清洗图片描述，添加头尾标志: %d ' % len(mapping))
    print(mapping['997722733_0cb5439472'])


    cap_max_length =  max_length(mapping)
    print('标题最大长度: %d ' % cap_max_length)


    all_vocab = to_vocabulary(mapping)
    print('生成词汇表: %d' % len(all_vocab))

    cut_vocab = cut_vocabulary(mapping)
    print('精简词汇表，词频高于10: %d' % len(cut_vocab))

    filename = 'dataset/Pickle/simple_dictionary.pkl'
    # word2ix, ix2word = vocab_mapping(cut_vocab,filename)
    # print('词汇映射表: %d' % len(word2ix))
    # print('startseq,endseq: %d,%d' % (word2ix['startseq'],word2ix['endseq']) )

    word2ix2, ix2word2 = loac_dictionary(filename)
    print('startseq,endseq: %d,%d' % (word2ix2['startseq'], word2ix2['endseq']))
    print('startseq,endseq: %d' % (word2ix2.get('croquet',-1)))
