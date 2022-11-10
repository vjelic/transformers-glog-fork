import sys
import numpy as np
import jsonlines as jsonl
from transformers import (
    GPT2Tokenizer, 
    GPT2TokenizerFast, 
    TFGPT2LMHeadModel, 
    GPT2LMHeadModel, 
    Trainer, 
    DataCollatorForLanguageModeling, 
    HfArgumentParser, 
    TrainingArguments, 
    )
import torch
import evaluate 
import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional

try:
    import tensorflow as tf
    from tensorflow.keras import metrics
except:
    pass

import random

@dataclass
class ModelArguments:
    batch_size: Optional[int] = field(default = 16)
    # We can't use TrainingArguments.fp16 because of internal TrainingArguments validity checking logic,
    # which throws an exception upon seeing --fp16 unless the PT install is GPU-enabled (even if we 
    # don't intend to use PT).
    tf_fp16: Optional[int] = field(default = 0)
    model_size: Optional[str] = field(default="Small")
    data_dir: Optional[str] = field(default="/data/tf-gpt-2/data/")

#data_dir="/data/tf-gpt-2/data"

def text_to_datasets(data_dir):
    train_json=data_dir+"/custom.train.jsonl"
    test_json=data_dir+"/custom.test.jsonl"
    try:
       f=open(train_json, "r")
       f.close()
       return
    except:
       print("Training JSONs missing, trying to generate...")
       pass

    try:
      # original: https://ymarkov.livejournal.com/280578.html
      f1=open(data_dir+,"r").readlines()
      f1=' '.join(f1)
      # original: https://archiveofourown.org/works/13291395 (postprocessed by removing HTML 
      # tags and non-ASCII characters)
      f=open(data_dir+"/gravity_falls.txt","r").readlines()
      f=' '.join(f)
    except:
      print("Source texts missing! Please download and try again\n"
      "Save https://drive.google.com/file/d/1F7FuipSI8VFqMnDmH3d9Bh3g2pI2BLAU/view?usp=sharing as "+data_dir+"/ringbearer_ascii.txt\n"
      "Save https://drive.google.com/file/d/14w8jtO1tjZWBVQ6fg7Z_xzhvDt2rnLR8/view?usp=sharing as "+data_dir+"/gravity_falls.txt\n"
      )
      exit(0)

    f=f+f1
    f=f.replace('\n','').replace('  ',' ').replace('  ',' ')

    of1 = open(train_json, "w")
    of2 = open(test_json, "w")
    ct1 = 1
    ct2 = 1
    ds1=[]
    ds2=[]
    import json
    row = 0
    for x in range(0, len(f)-4096-1, 256):
        startpos = x
        endpos = x+4096
        if x>0:
            while f[startpos-1]!=' ':
                startpos+=1
        while f[endpos]!=' ':
            endpos-=1
        fragment = f[startpos:endpos].replace('"', '\\"')
        if (row % 10):
            #of1.write('{"id": %d, "text": "%s", "length": %d, "ended": false}\n' % (ct1, fragment, endpos-startpos))
            json.dump({"id":ct1, "text":fragment, "length":endpos-startpos, "ended":False}, of1)
            of1.write('\n')
            ct1+=1
        else:
            #of2.write('{"id": %d, "text": "%s", "length": %d, "ended": false}\n' % (ct2, fragment, endpos-startpos))
            json.dump({"id":ct2, "text":fragment, "length":endpos-startpos, "ended":False}, of2)
            of2.write('\n')
            ct2+=1
        row += 1
    of1.close()
    of2.close()


# An explicit way to tokenize: a simplified version of what DataCollatorForLanguageModeling does
def tokenize(tokenizer, lines):
    """
    try:
        valid_dataset = np.loadz("dataset_valid.npz")
        xs = valid_dataset['xs']
        ys = valid_dataset['ys']
        valid_dataset = tf.data.Dataset.from_tensor_slices((xs, tf.constant(ys)))
    except:
        xs, ys, valid_dataset = tokenize(tokenizer,400)
        np.savez("dataset_valid.npz", xs=xs, ys=ys)

    try:
        train_dataset = np.loadz("dataset_train.npz")
        xs = valid_dataset['xs']
        ys = valid_dataset['ys']
        train_dataset = tf.data.Dataset.from_tensor_slices((xs, tf.constant(ys)))
    except:
        xs, ys, train_dataset = tokenize(tokenizer,16000)
        np.savez("dataset_train.npz", xs=xs, ys=ys)
    """
    f1=open("/dockerx/ringbearer_ascii.txt","r").readlines()
    f1=' '.join(f1)
    f=open("/dockerx/gravity_falls.txt","r").readlines()
    f=' '.join(f)
    f=f+f1
    data=f.replace('\n','').replace('  ',' ').replace('  ',' ')
    data = tokenizer.encode(data, return_tensors='tf', padding=True, truncation=False)
    hash_token = tokenizer(['#'], return_tensors='tf')
    hash_token = hash_token['input_ids'][0,0]

    input_ids=[]
    masks=[]
    responses=[]
    for k in range(lines):
        pos = random.randint(0, len(data[0])-1024-1)
        query = data[0][pos:pos+256].numpy()
        attention_mask=np.ones([256], dtype=np.int32)
        response = query.copy()

        # Intervention #1
        attention_mask[255]=0
        query[255]=hash_token
        response[255] = -100

        # Intervention #2
        mask = np.random.binomial(1,0.15,[256])
        query = np.where(mask, hash_token, query)
        response = np.where(mask, response, -100)
        attention_mask = attention_mask*(1-mask)
        #print(mask, response)
        input_ids.append(query)
        masks.append(attention_mask)
        responses.append(response)

    xs = {'input_ids': np.stack(input_ids), 'attention_mask': np.stack(masks), "labels": np.stack(responses)}
    ys = np.stack(responses)
    return xs, ys, tf.data.Dataset.from_tensor_slices((xs, tf.constant(ys)))

acc_metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    mask = (labels!=-100)
    result = acc_metric.compute(predictions=preds[mask], references=labels[mask])
    return result

# not used at present
class MaskedCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self):
        super().__init__(from_logits=True)

    def call(self, y_true, y_pred):
        mask = y_true != -100
        return super().call(y_true[mask], y_pred[mask])

class MaskedAccuracy(metrics.SparseCategoricalAccuracy):
    def __init__(self):
        super().__init__(name="Accuracy")
        self.crossent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:,1:]
        y_pred = y_pred[:,:-1,:]
        mask = y_true != -100
        return super().update_state(y_true[mask], y_pred[mask])

def model_test(model, tokenizer, is_tf=True):
    prefix = ""
    prompts = [
        "it was a dark and stormy",
        "feet are lifted toward the heavens and the jade stalk is",
        "though it was once forested, Ithilien has become",
        "Dipper has a twin sister, and her name is",
        "His Majesty Sauron has declared recently"
    ]
    for prompt_text in prompts:
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="tf" if is_tf else "pt")
        if not is_tf:
            encoded_prompt=encoded_prompt.to(torch.device("cuda:0"))
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            min_length=10 + len(encoded_prompt[0]),
            max_length=50 + len(encoded_prompt[0]),
            temperature=1.0,
            top_k=0.0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=5,
            attention_mask = None,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )
            generated_sequences.append(total_sequence)
            print(total_sequence)


def init_datasets(model_args, tokenizer):
    def preprocess_function(examples):
        rv=tokenizer(examples["text"], padding='max_length', truncation=True, max_length=1024)
        return {"input_ids": rv["input_ids"], "attention_mask":rv["attention_mask"]}

    train_file = model_args.data_dir + "custom.train.jsonl"
    valid_file = model_args.data_dir + "custom.test.jsonl"

    pt_dataset = load_dataset("json", data_files={"train":train_file, "validation":valid_file, "test":valid_file})
    tokenized_dataset = pt_dataset.map(preprocess_function, num_proc=16, remove_columns=["id", "text", "ended", "length"])
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"].select(range(384 if model_args.tf_flag else 96))
    if not model_args.tf_flag:
        return train_dataset, eval_dataset
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
    tf_train_dataset = train_dataset.to_tf_dataset(
        # labels are passed as input, as we will use the model's internal loss
        columns=["labels", "input_ids", "attention_mask"],
        shuffle=True,
        batch_size=model_args.batch_size,
        collate_fn=data_collator,
        drop_remainder=True,
    ).with_options(options)

    tf_eval_dataset = eval_dataset.to_tf_dataset(  
        columns=["labels", "input_ids", "attention_mask"],
        shuffle=True,
        batch_size=model_args.batch_size,
        collate_fn=data_collator,
        drop_remainder=True,
    ).with_options(options)
    return tf_train_dataset, tf_eval_dataset


def test_tf(model_args, training_args):
    global tokenizer
    devices = []

    gpus = tf.config.experimental.list_physical_devices('GPU')
    #for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)

    for i in range(len(gpus)):
        devices.append("GPU:"+str(i))
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print("learning rate", training_args.learning_rate)
    print("FP16 mode", model_args.tf_fp16)
    with strategy.scope():
        tokenizer = GPT2TokenizerFast.from_pretrained(model_args.model_name, mask_token='#')
        tokenizer.pad_token = tokenizer.eos_token

        train_dataset, eval_dataset = init_datasets(model_args, tokenizer)
        #train_dataset = train_dataset.repeat()
        try:
            model = TFGPT2LMHeadModel.from_pretrained(model_args.model_size+"-pretrained")
        except:
            model = TFGPT2LMHeadModel.from_pretrained(model_args.model_name)
        #model = TFGPT2LMHeadModel.from_pretrained(model_args.model_name)
        model.config.use_cache=False
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate)
        if model_args.tf_fp16:
            if model_args.tf_fp16==1:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True, initial_scale=256)
            else:
                optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic')
        loss = MaskedCrossentropy()
        #metric = MaskedAccuracy()
        #metric.tokenizer=tokenizer
        #model.compile(optimizer=optimizer, loss="passthrough", metrics=[metric])
        #model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        model.compile(optimizer=optimizer, loss="passthrough", metrics=[])

        #model_test(model, tokenizer, is_tf=True)
        print("====== TRAINING ======")
        acc=0
        for k in [1,1,3,5,5,5]:
            model.fit(train_dataset, batch_size=1, epochs=k)
            model.save_pretrained(model_args.model_size+"-pretrained")
            acc+=k
            print(f"====== After {acc} epoch: ======")
            model_test(model, tokenizer, is_tf=True)
            model.evaluate(eval_dataset)

def test_pt(model_args, training_args):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_args.model_name, mask_token='#')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset, eval_dataset = init_datasets(model_args, tokenizer)
    pt_model = GPT2LMHeadModel.from_pretrained(model_args.model_name).to(torch.device("cuda"))
    pt_model.config.use_cache=False
    #pt_model.bfloat16()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
    # For some reason, I run out of GPU memory if I go any higher than 4x1 and eval dataset length 96
    training_args.per_device_train_batch_size=4
    training_args.per_device_eval_batch_size=1
    training_args.max_steps=3000
    #training_args.evaluation_strategy="epoch"
    print("========================= Evaluation on the PT model (huggingface checkpoint) ==================================")
    trainer = Trainer(
            model=pt_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train(resume_from_checkpoint=False)
    if training_args.local_rank==0:
      model_test(pt_model, tokenizer, is_tf=False)
    status=trainer.evaluate()
    if training_args.local_rank==0:
      print(training_args.local_rank,status)

def main():
    np.set_printoptions(linewidth=250)
    tf_flag = sys.argv[1]=='tf' or sys.argv[2]=='tf'
    if sys.argv[2]=='pt' or sys.argv[2]=='tf':
        sys.argv = sys.argv[:2]+sys.argv[3:]
    else:
        sys.argv = sys.argv[:1]+sys.argv[2:]
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.                                       
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.local_rank is None:
        training_args.local_rank = 0
    if model_args.model_size == "Small":
        model_args.model_name = "gpt2"
    elif model_args.model_size == "Medium":
        model_args.model_name = "gpt2-medium"
    elif model_args.model_size == "Large":
        model_args.model_name = "gpt2-large"
    elif model_args.model_size == "XL":
        model_args.model_name = 'gpt2-xl'
    model_args.tf_flag = tf_flag
    text_to_datasets(model_args.data_dir)
    if tf_flag:
        test_tf(model_args, training_args)
    else:
        test_pt(model_args, training_args)

if __name__ == "__main__":
    main()
