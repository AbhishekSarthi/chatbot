from flask import Flask , request,jsonify
import numpy as np
import json
from keras.models import load_model

app = Flask(__name__)

encoding = np.load('enc_list(1).npy')
print(encoding.shape)

with open('word_dict(1).json', 'r') as openfile:
    word_dict = json.load(openfile)
# print(word_dict)
print(type(word_dict))

with open('word_list(1).json', 'r') as openfile:
    word_list = json.load(openfile)
# print(word_list)
print(len(word_list))

word_enc={}
for i in range(0,len(word_list)):
    word_enc[word_list[i]] = encoding[i]

model = load_model('chat_model(1).h5')    


def decode(ind,word_dict):
    ind = str(ind)
    ans = word_dict[ind]
    #print(ans)
    return ans

def second_max_index(row,first):
    maxi = -1
    ind=0
    count=0
    for i in row:
        if i>maxi and i<first:
            maxi = i
            ind =count
        count =count+1
    #print('ind = ',ind,maxi)
    return ind,maxi    

def max_index(row):
    maxi = -1
    ind=0
    count=0
    for i in row:
        if i>maxi:
            maxi = i
            ind =count
        count =count+1
    #print('ind = ',ind,maxi)
    return ind,maxi

def encode_sent(sent,word_enc):
    lmax =10
    sent = sent.split(" ")
    dif = lmax - len(sent)
    print(dif)
    while dif>1:
        dif = lmax - len(sent)
        sent.append(" ")
    enc = []
    print(sent)
    for i in sent:
        enc.append(word_enc[i])
    enc = np.array(enc)
    print(enc.shape)
    return enc


def pred(model,sample,word_dic,word_enc):
    sample = encode_sent(sample,word_enc)
    print(sample.shape)
    use=[]
    words=[]
    for i in sample:
        index = np.argmax(i,axis=0)
        ans = decode(str(index),word_dic)
        use.append(ans)
    sample = sample.reshape(1,sample.shape[0],sample.shape[1])
    print(sample.shape)
    y = model.predict(sample)
    print(y)
    #for i in sample[0]:
    #  a = i.reshape(1,sample[0].shape[1])
    #  yw = modelw.predict(a)
    #  index = np.argmax(yw,axis=1)
    #  ans = decode(index[0],word_dic)
    #  words.append(ans)
    y = y.reshape(y.shape[1],y.shape[2])
    print(y)
    indice=[]
    index=[]
    sen=[]
    sen1 = []
    sen2=[]
    sen3=[]
    for i in y:
        #print(i)
        #print(i.shape)
        ind,maxi = max_index(i)
        indice.append(ind)
        ans = decode(ind,word_dic)
        sen1.append(ans)
        ind,maxi = second_max_index(i,maxi)
        indice.append(ind)
        ans = decode(ind,word_dic)
        sen2.append(ans)
        ind,maxi = second_max_index(i,maxi)
        indice.append(ind)
        ans = decode(ind,word_dic)
        sen3.append(ans)
    print(use)
    print(sen1)
    print(sen2)
    print(sen3)
    #print(words)

# a = encode_sent("how are you",word_enc)        

# pred(model,"hi",word_dict,word_enc)

@app.route('/')
def index():
	return 'Hello'


@app.route('/pred', methods=['GET', 'POST'])
def prediction():
	if request.method == 'POST':
		print(request.json["data"])
		pred(model,request.json["data"],word_dict,word_enc)
		return request.data
	return 'hellp'	
      

if __name__ == "__main__":
	app.run()	
