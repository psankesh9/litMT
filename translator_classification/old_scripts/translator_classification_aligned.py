import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datasets, evaluate, pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import wandb
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from transformers import TrainerCallback


## CONFIGS
GPU = "1"
PROJ_PATH = '/home/kkatsy/litMT'
EXP_NAME = 'random_src_tgt'
TRAIN_SET = 'random_train_df.pickle'
WANDB_PROJ = 'translator-classification-exp-lr2e7-wd0.5'

SAVE_PATH = PROJ_PATH + '/' + EXP_NAME
OUT_PATH = PROJ_PATH + '/exp_logs_lr2e7_wd0.5/' + EXP_NAME
PRETRAINED_PATH = '/home/kkatsy/pretrained'

BERT_MODEL = "bert-base-multilingual-cased"
FREEZE_BERT = False
UNFREEZE_LAYER = None
FINE_TUNE = True
LOAD_TUNED = False

lr = 2e-7
epochs = 25
batch_size = 12


if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
    
if not os.path.exists("/projects/kkatsy/" + EXP_NAME):
    os.makedirs("/projects/kkatsy/" + EXP_NAME)  
    
if not os.path.exists("/projects/kkatsy/"+ PROJ_PATH):
    os.makedirs("/projects/kkatsy/"+ PROJ_PATH) 
    
if not os.path.exists("/projects/kkatsy/"+ PROJ_PATH + '/' + EXP_NAME):
    os.makedirs("/projects/kkatsy/"+ PROJ_PATH + '/' + EXP_NAME) 
    
# GPU SETUP
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


# CLASS LABELS
label_list = ["Garnett", "McDuff", "PV", "Katz", "Hogarth"]
le = LabelEncoder()
le.fit(label_list)
id_list = le.transform(list(label_list))

id2label, label2id = {}, {}
for l, i in zip(label_list, id_list):
    id2label[i] = l
    label2id[l] = i
    
    
# PREP MODEL + TRAINER ELEMS
if FINE_TUNE: 
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(label_list))
    
    if FREEZE_BERT:
        print('bert\'s been frozen')
        for param in model.bert.parameters():
            param.requires_grad = False
    
    model.to(device) 
elif LOAD_TUNED:
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_PATH)
    model.to(device)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
class FreezingCallback(TrainerCallback):

    def __init__(self, unfreezing_epoch: int, trainer: Trainer):
        self.trainer = trainer
        self.unfreezing_epoch = unfreezing_epoch
        self.is_unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch >= self.unfreezing_epoch:
            if not self.is_unfrozen:
                self.unfreeze_model(int(state.epoch))
                self.is_unfrozen = True

    def unfreeze_model(self, epoch: int):
        print('Unfreezing model at epoch ', epoch)
        for param in self.trainer.model.bert.parameters():
            param.requires_grad = True

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(datum):
    src, tgt = datum['text'].split(' <SEP> ')
    return tokenizer(src, tgt, padding='max_length', max_length=512, truncation='longest_first')


## LOAD + PREP DATA
train_df = pd.read_pickle(PROJ_PATH + "/experiment_dataset/" + TRAIN_SET)  
test_df = pd.read_pickle(PROJ_PATH + "/experiment_dataset/experiment_test_df.pickle") 
val_df = pd.read_pickle(PROJ_PATH + "/experiment_dataset/experiment_val_df.pickle") 

sentences = {}
sentences['train'] = [{'label': row['labels'], 'text':row['concat']} for i, row in train_df.iterrows()]
sentences['test'] = [{'label': row['labels'], 'text':row['concat']} for i, row in test_df.iterrows()]
sentences['val'] = [{'label': row['labels'], 'text':row['concat']} for i, row in val_df.iterrows()]

train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['train']))
val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['val']))
test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['test']))

tokenized_train = train_dataset.map(preprocess_function)
tokenized_val = val_dataset.map(preprocess_function)
tokenized_test = test_dataset.map(preprocess_function)


# START-UP WANDB
run = wandb.init(
        # Set the project where this run will be logged
        project=WANDB_PROJ,
        name = EXP_NAME,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )
os.environ["WANDB_PROJECT"]=WANDB_PROJ


# TRAIN || LOAD MODEL
if FINE_TUNE:
    training_args = TrainingArguments(
        output_dir="/projects/kkatsy/" + EXP_NAME,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.05,
        warmup_steps=5000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
        logging_dir="/home/kkatsy/litMT/logs/",
        logging_strategy="epoch",
        logging_first_step=False,
        lr_scheduler_type='linear'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.add_callback(CustomCallback(trainer))
    if FREEZE_BERT:
        freezing_callback = FreezingCallback(UNFREEZE_LAYER, trainer)
        trainer.add_callback(freezing_callback)
    
    trainer.train()
    trainer.evaluate()
    
    eval_accuracy = []
    eval_loss = []
    train_loss = []
    train_accuracy = []
    for d in trainer.state.log_history:
        # run.log(d)
        if 'eval_loss' in d.keys():
            eval_loss.append(d['eval_loss'])
            eval_accuracy.append(d['eval_accuracy'])
        elif 'train_accuracy' in d.keys():
            train_loss.append(d['train_loss'])
            train_accuracy.append(d['train_accuracy'])
            
    run_results = {'train_loss': train_loss, 'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy}
    run.log(run_results)

    with open(OUT_PATH + '/run_results.pickle', 'wb') as handle:
        pickle.dump(run_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif LOAD_TUNED:
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.evaluate()
        

## GET MODEL PREDICTIONS EVAL
predictions, labels, metrics = trainer.predict(tokenized_test, metric_key_prefix="predict")

preds = np.argmax(predictions, axis=-1)
pred_true = {'pred': preds, 'true': list(test_dataset['label'])}

with open(OUT_PATH + '/pred_true.pickle', 'wb') as handle:
    pickle.dump(pred_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
translators = [id2label[l] for l in preds]
pred_count = {}
for i in label2id.keys():
    count = translators.count(i)
    pred_count[i] = count
    
translators = list(pred_count.keys())
true_count = {}
for i in id2label.keys():
    count = list(test_dataset['label']).count(i)
    true_count[id2label[i]] = count
    
predicted = list(pred_count.values())
actual = list(true_count.values())
X_axis = np.arange(len(label_list))
  
plt.bar(X_axis - 0.2, actual, 0.4, label = 'TRUE', color='firebrick') 
plt.bar(X_axis + 0.2, predicted, 0.4, label = 'PRED', color='steelblue') 
  
plt.xticks(X_axis, label_list) 
plt.xlabel("Translators") 
plt.ylabel("Translator Par Counts") 
plt.title("Predicted vs True Translator Distribution") 
plt.legend() 
plt.show() 
plt.savefig(OUT_PATH + '/true_actual_dist.png')

acc = accuracy_score(labels, preds)
run.log({"Test accuracy": acc})
print('Test Accuracy: ', acc)

confusion_matrix = confusion_matrix(labels, preds, normalize='pred')
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=label2id)
cm_display.plot()
plt.show()
plt.savefig(OUT_PATH + '/confusion_matrix.png')


## CLOSE WANDB
wandb.finish()