from ViNLP.BertPosTagger import BERTPoSTagger
from transformers import BertTokenizer
from transformers import  tokenization_bert
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
import torch
import re
from tqdm import tqdm
import unicodedata
import logging
import os
import unicodedata
import zipfile
import urllib.request
logger = logging.getLogger(__name__)
def _is_punctuation(char):
    return False
tokenization_bert._is_punctuation = _is_punctuation

if torch.cuda.is_available():       
    device = torch.device("cuda")

    logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())

    logger.info('We will use the GPU:' % torch.cuda.get_device_name(0))
else:
    logger.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class BertVnNer:
    def __init__(self,model_path=None,max_length=256,tag2int=None):
        if(tag2int is None):
            self.tag2int = {'B-LOCATION+B-ORGANIZATION': 1, 'B-LOCATION+I-LOCATION': 2, 'B-LOCATION+I-MISCELLANEOUS': 3, 'B-LOCATION+I-ORGANIZATION': 4, 
                            'B-LOCATION+O': 5, 'B-MISCELLANEOUS+I-ORGANIZATION': 6, 'B-MISCELLANEOUS+O': 7, 'B-ORGANIZATION+B-LOCATION': 8, 'B-ORGANIZATION+I-ORGANIZATION': 9, 
                            'B-ORGANIZATION+O': 10, 'B-PERSON+B-ORGANIZATION': 11, 'B-PERSON+I-LOCATION': 12, 'B-PERSON+I-ORGANIZATION': 13, 'B-PERSON+O': 14, 'I-LOCATION+I-LOCATION': 15, 
                            'I-LOCATION+I-MISCELLANEOUS': 16, 'I-LOCATION+I-ORGANIZATION': 17, 'I-LOCATION+O': 18, 'I-MISCELLANEOUS+I-ORGANIZATION': 19, 'I-MISCELLANEOUS+O': 20,
                            'I-ORGANIZATION+I-ORGANIZATION': 21, 'I-ORGANIZATION+O': 22, 'I-PERSON+I-LOCATION': 23, 'I-PERSON+I-ORGANIZATION': 24, 'I-PERSON+O': 25, 'O+B-LOCATION': 26, 
                            'O+B-MISCELLANEOUS': 27, 'O+B-ORGANIZATION': 28, 'O+I-LOCATION': 29, 'O+I-MISCELLANEOUS': 30, 'O+I-ORGANIZATION': 31, 'O+O': 32, '-PAD-+-PAD-': 0}
        else:
            self.tag2int = tag2int
        self.int2tag = {v: k for k, v in self.tag2int.items()}
        if(model_path is None): 
            path_root = os.path.join(os.path.expanduser('~'),".cache/torch/transformers")
            if(os.path.exists(os.path.join(path_root,"VnBertNer"))):
                model_path = os.path.join(path_root,"VnBertNer")
                logger.info("Load model from cache : {}".format(os.path.join(path_root,"VnBertNer")))
            else:
                if(not os.path.exists(path_root)):
                    os.makedirs(path_root)
                logger.info("Downloading.... model from https://insai.s3-ap-southeast-1.amazonaws.com/transformers_model/VnBertNer.zip")
                urllib.request.urlretrieve('https://insai.s3-ap-southeast-1.amazonaws.com/transformers_model/VnBertNer.zip', os.path.join(path_root,'VnBertNer.zip'))
                logger.info("Model is saved in {}".format(path_root))
                with zipfile.ZipFile(os.path.join(path_root,'VnBertNer.zip'), 'r') as zip_ref:
                    zip_ref.extractall(path_root)
                model_path = os.path.join(path_root,'VnBertNer')
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False,tokenize_chinese_chars=False)
        self.bertPoSTagger = BERTPoSTagger.from_pretrained(
                                            model_path,
                                            num_labels = len(self.tag2int),
                                            output_attentions = False,
                                            output_hidden_states = False,
                                    )
        self.MAX_LENGTH= max_length
        if torch.cuda.is_available():
            self.bertPoSTagger.cuda()

    def covert_text(self,texts,batch_size):
        self.sents_tok = []
        input_ids = []
        for i  in range(len(texts)):
            sent = texts[i]
            self.sents_tok.append(self.tokenizer.tokenize(sent)[0:self.MAX_LENGTH-2])
            encoded_sent = self.tokenizer.encode(
                                    sent,                      # Sentence to encode.
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    #return_tensors = 'pt',     # Return pytorch tensors.
                        )
            if(len(encoded_sent)<= self.MAX_LENGTH):
                encoded_sent= encoded_sent + [0]*(self.MAX_LENGTH-len(encoded_sent))
            else:
                encoded_sent = encoded_sent[0:self.MAX_LENGTH-1] + [encoded_sent[-1]]
            input_ids.append(encoded_sent)

        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        train_inputs = input_ids
        train_masks= attention_masks
        train_inputs = torch.tensor(train_inputs,dtype=torch.long)
        train_masks = torch.tensor(train_masks,dtype=torch.long)
        train_data = TensorDataset(train_inputs, train_masks)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return train_dataloader

    def preprocess(self,texts):
        new_texts = []
        for t in texts:
            s = " ".join(nltk.word_tokenize(t.strip()))
            new_texts.append(s)
        return new_texts

    def predict(self,dataLoader):
        self.bertPoSTagger.eval()
        test_pred= []
        for step, batch in enumerate(tqdm(dataLoader, desc="Predict")):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():        
                outputs =  self.bertPoSTagger(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
                    
            predictions = outputs[0]
            max_preds = predictions.argmax(dim = 2, keepdim = True)
            lb = max_preds.cpu().detach().numpy().tolist()
            for i in range(len(lb)):
                results = []
                sent_tok = self.sents_tok[step*self.batch_size+i]
                sent_ori = self.texts[step*self.batch_size+i].split()
                idx= 0
                for w in range(0,len(sent_tok)):
                    if(sent_tok[w]=="[UNK]"):
                        try:
                            word = sent_ori[idx]
                        except:
                            logging.info(w)
                            logging.info(sent_tok)
                            logging.info(sent_ori)
                        results.append([word,self.int2tag[lb[i][w+1][0]].split("+")[0]])
                        idx = idx + 1
                    elif("##" in sent_tok[w]):
                        word = sent_tok[w]
                        results[-1][0]= results[-1][0] + word.replace("##","")
                    else:
                        word = sent_tok[w]
                        results.append([word,self.int2tag[lb[i][w+1][0]].split("+")[0]])
                        idx = idx + 1
                test_pred.append(results)
        return test_pred

    def split(self,texts,batch_size=16):
        texts = self.preprocess(texts)
        self.texts = texts
        self.batch_size = batch_size
        train_dataloader = self.covert_text(texts,batch_size)
        test_pred = self.predict(train_dataloader)
        return test_pred

    def annotate(self,texts,batch_size=8):
        test_pred = self.split(texts,batch_size=batch_size)
        entities = []
        for sent in test_pred:
            tmp= {}
            word= ""
            tag = ""
            for w in sent:
                if(w[1]!='O'):
                    if('B' in w[1]):
                        word = ""
                        tag = ""
                        word = word + w[0]
                        tag = w[1].split("-")[1]
                    elif('I' in w[1]):
                        word = word + " "+w[0]
                else:
                    if(word!=""):
                        if tag not in tmp:
                            tmp[tag]=[word]
                        else:
                            if(word not in tmp[tag]):
                                tmp[tag].append(word)
                        word = ""
                        tag = ""
            entities.append(tmp)
        return entities
