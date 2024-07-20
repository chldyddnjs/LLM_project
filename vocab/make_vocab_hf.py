import os
import argparse
from datasets import load_dataset
from tokenizers import (
    BertWordPieceTokenizer,
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    )

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path',type=str,required=True)
    parser.add_argument('--vocab_size',type=int,default=32000,required=True)
    parser.add_argument('--limit_alphabet', type=int, default=0)
    parser.add_argument('--tokenizer',type=str, default='sentencepiece')
    parser.add_argument('--model_type',type=str,default='gpt')
    parser.add_argument('--save_dir',type=str,default='data')
    parser.add_argument('--min_frequency',type=int,default=5)
    parser.add_argument('--split',type=str,default="train[:]")
    return parser.parse_args()

def get_training_corpus(corpus_path:str,batch_size:int=5000,split:str="train[:]"):

    if ',' in corpus_path:
        path,data_dir = corpus_path.split(',')
        print(path,data_dir)
        dataset = load_dataset(path,data_dir,num_proc=os.cpu_count()-1,split=split)
    else:
        dataset = load_dataset(corpus_path,num_proc=os.cpu_count()-1,split=split)
    
    print(f"Load the {corpus_path} datasets")
    print('length of datasets : {0}'.format(len(dataset['text'])))
    print("Check one example :",dataset['text'][0])
    
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]['text']

def main(args):

    if ',' in args.corpus_path:
        path,data_dir = args.corpus_path.split(',')
        dataset = load_dataset(path,data_dir,num_proc=os.cpu_count()-1)
    else:
        dataset = load_dataset(args.corpus_path,num_proc=os.cpu_count()-1)
    if args.model_type == 'bert':
        special_tokens = ['<|sos|>',
                          '<|eos|>',
                          '<|pad|>',
                          '<|unk|>',
                          '<|mask|>',
                          '<|sep|>',
                          '<|cls|>']
    elif args.model_type == 'gpt':
        special_tokens=[
            '<s>',
            '</s>',
            '<unk>',
        ]
    else:
        raise ValueError(f"Unsupported model type:",{args.model_type})

    if args.tokenizer.lower() == 'wordpiece':
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
            wordpieces_prefix='##'
        )

    elif args.tokenizer.lower() == 'bbpe':
        tokenizer = ByteLevelBPETokenizer(
            lowercase=False,
            add_prefix_space=True,
            unicode_normalizer='nfc'
        )
    elif args.tokenizer.lower() == 'char-bpe':
        tokenizer = CharBPETokenizer(
            lowercase=False,
            add_prefix_space=True,
            unicode_normalizer='nfc',
        )
    elif args.tokenizer.lower() == 'sentencepiece':
        tokenizer = SentencePieceBPETokenizer()
    else:
        raise ValueError("Not found tokenizer!")

    tokenizer.train_from_iterator(
            get_training_corpus(args.corpus_path,split=args.split),
            vocab_size = args.vocab_size,
            min_frequency = args.min_frequency, 
            show_progress = True,
            special_tokens = special_tokens,
        )

    if not os.path.exists(args.save_dir + '_' + args.tokenizer):
        os.mkdir(args.save_dir + '_' + args.tokenizer)
    

    tokenizer.save_model(f'./{args.save_dir}_{args.tokenizer}')

if __name__=="__main__":
    args = setup_args()
    main(args)
