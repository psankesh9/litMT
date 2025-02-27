import torch, datasets, transformers
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import io
import wandb
transformers.logging.set_verbosity_error()

device = torch.device('cuda')

def get_labels(classes):
    le = LabelEncoder()
    le.fit(classes)
    id_list = le.transform(list(classes))

    id2label, label2id = {}, {}
    for l, i in zip(classes, id_list):
        id2label[i] = l
        label2id[l] = i
    
    return id2label, label2id

def to_torch(dataset):
    torchified = dataset.rename_column('label', 'labels')
    torchified.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return torchified


def preprocess_data(model, train_file, val_file, test_file, context_type, batch_size):
    
    train_df = pd.read_pickle(train_file)  
    val_df = pd.read_pickle(val_file) 
    test_df = pd.read_pickle(test_file) 

    sentences = {}
    sentences['train'] = [{'label': row['labels'], 'text':row['concat']} for i, row in train_df.iterrows()]
    sentences['test'] = [{'label': row['labels'], 'text':row['concat']} for i, row in test_df.iterrows()]
    sentences['val'] = [{'label': row['labels'], 'text':row['concat']} for i, row in val_df.iterrows()]

    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['train']))
    val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['val']))
    test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=sentences['test']))
    
    tokenizer = BertTokenizer.from_pretrained(model)
    
    def tokenize_tgt256(datum):
        src, tgt = datum['text'].split(' <SEP> ')
        
        return tokenizer(tgt, padding='max_length', 
                        max_length=256, 
                        truncation='longest_first',
                        add_special_tokens=True
                        )
    def tokenize_tgt512(datum):
        src, tgt = datum['text'].split(' <SEP> ')
        
        return tokenizer(tgt, 
                        padding='max_length', 
                        max_length=512, 
                        truncation='longest_first',
                        add_special_tokens=True
                        )
    def tokenize_src_tgt(datum):
        src, tgt = datum['text'].split(' <SEP> ')
        
        return tokenizer(src, tgt, 
                        padding='max_length', 
                        max_length=512, 
                        truncation='longest_first',
                        add_special_tokens=True
                        )
        
    if context_type == "tgt256":
        tokenized_train = train_dataset.map(tokenize_tgt256)
        tokenized_val = val_dataset.map(tokenize_tgt256)
        tokenized_test = test_dataset.map(tokenize_tgt256)
    elif context_type == "tgt512":
        tokenized_train = train_dataset.map(tokenize_tgt512)
        tokenized_val = val_dataset.map(tokenize_tgt512)
        tokenized_test = test_dataset.map(tokenize_tgt512)
    elif context_type == "src+tgt":
        tokenized_train = train_dataset.map(tokenize_src_tgt)
        tokenized_val = val_dataset.map(tokenize_src_tgt)
        tokenized_test = test_dataset.map(tokenize_src_tgt)
    
    tokenized_train = to_torch(tokenized_train)
    tokenized_val = to_torch(tokenized_val)
    tokenized_test = to_torch(tokenized_test)
    
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_test, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader

def run_eval(model, the_dataloader):
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(the_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
                
            input_id_tensors = batch['input_ids']
            input_mask_tensors = batch['attention_mask']
            label_tensors = batch['labels']

            b_input_ids = input_id_tensors.to(device)
            b_input_mask = input_mask_tensors.to(device)
            b_labels = label_tensors.to(device)
            
            with torch.no_grad():        

                # forward pass
                outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                loss = outputs.loss
                logits = outputs.logits
                    
                total_eval_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                num_correct = np.sum(pred_flat == labels_flat)
                total_correct += num_correct
                total_samples += batch['labels'].size(0)
            
        avg_acc = total_correct / total_samples
        avg_loss = total_eval_loss / len(the_dataloader)

        return avg_loss, avg_acc

def get_preds(model, the_dataloader):
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        total_correct = 0
        total_samples = 0
        predictions = []
        labels = []
        
        for batch in tqdm(the_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
                
            input_id_tensors = batch['input_ids']
            input_mask_tensors = batch['attention_mask']
            label_tensors = batch['labels']
            
            b_input_ids = input_id_tensors.to(device)
            b_input_mask = input_mask_tensors.to(device)
            b_labels = label_tensors.to(device)
            
            with torch.no_grad():        

                # forward pass
                outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                loss = outputs.loss
                logits = outputs.logits
                    
                total_eval_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                num_correct = np.sum(pred_flat == labels_flat)
                total_correct += num_correct
                total_samples += batch['labels'].size(0)
                predictions.extend(list(pred_flat))
                labels.extend(list(labels_flat))

        avg_acc = total_correct / total_samples
        avg_loss = total_eval_loss / len(the_dataloader)
    
        return avg_acc, avg_loss, predictions, labels

def get_plots(run, classes, labels, preds):
    
    id2label, label2id = get_labels(classes)
    
    translators = [id2label[l] for l in preds]
    pred_count = {}
    for i in label2id.keys():
        count = translators.count(i)
        pred_count[i] = count
        
    translators = list(pred_count.keys())
    true_count = {}
    for i in id2label.keys():
        count = labels.count(i)
        true_count[id2label[i]] = count
        
    predicted = list(pred_count.values())
    actual = list(true_count.values())
    X_axis = np.arange(len(classes))
    
    plt.bar(X_axis - 0.2, actual, 0.4, label = 'TRUE', color='firebrick') 
    plt.bar(X_axis + 0.2, predicted, 0.4, label = 'PRED', color='steelblue') 
    
    plt.xticks(X_axis, classes) 
    plt.xlabel("Translators") 
    plt.ylabel("Translator Par Counts") 
    plt.title("Predicted vs True Translator Distribution") 
    plt.legend() 
    plt.show() 
    
    buf_dist = io.BytesIO()
    plt.savefig(buf_dist, format='png')
    buf_dist.seek(0)
    run.log({'predicted_distribution' : wandb.Image(Image.open(buf_dist))})
    print("Saved Prediction Distribution")
    
    conf_matrix = confusion_matrix(labels, preds, normalize='pred')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=label2id)
    cm_display.plot()
    plt.show()
    
    buf_conf = io.BytesIO()
    plt.savefig(buf_conf, format='png')
    buf_conf.seek(0)
    run.log({'confusion_matrix' : wandb.Image(Image.open(buf_conf))})
    print("Saved Confusion Matrix")

def fine_tune(model, classes, data, train_args, proj_name, run_name, model_save_pth):
    
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(classes))
    model.to(device)

    optimizer = AdamW(model.parameters(), 
                eps = train_args['eps'],
                lr = train_args['lr'],
                weight_decay = train_args['wd']) 
    
    run = wandb.init(project=proj_name, name=run_name)
    
    # Get results for Epoch 0
    print('Epoch 0: ')
    train_loss, train_acc = run_eval(model, data['train'])
    val_loss, val_acc = run_eval(model, data['val'])
    print(f"Train accuracy: {train_acc:.4f}, Train loss: {train_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")
    run.log({"train_accuracy": train_acc, 'train_loss': train_loss, "val_accuracy": val_acc, 'val_loss': val_loss})
    print("")

    for epoch in range(train_args['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, train_args['epochs']))
        print('Training...')

        total_train_loss = 0
        total_correct = 0
        total_samples = 0
        
        model.train()
        
        for batch in tqdm(data['train']):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_id_tensors = batch['input_ids']
            input_mask_tensors = batch['attention_mask']
            label_tensors = batch['labels']
            
            b_input_ids = input_id_tensors.to(device)
            b_input_mask = input_mask_tensors.to(device)
            b_labels = label_tensors.to(device)

            # clear gradients
            model.zero_grad()        

            # forward pass
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # calc correct logits
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            num_correct = np.sum(pred_flat == labels_flat)
            total_correct += num_correct
            total_samples += batch['labels'].size(0)

            total_train_loss += loss.item()

            # backward pass to calc gradients.
            loss.backward()

            # update params + take step using gradient
            optimizer.step()
            
        train_acc = total_correct / total_samples
        train_loss = total_train_loss / len(data['train'])
        val_loss, val_acc = run_eval(model, data['val'])
        
        print(f"Train accuracy: {train_acc:.4f}, Train loss: {train_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")
        run.log({"train_accuracy": train_acc, 'train_loss': train_loss, "val_accuracy": val_acc, 'val_loss': val_loss})
        
    # test dataset performance
    test_acc, test_loss, test_preds, test_labels = get_preds(model, data['test'])
        
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    run.log({"test_accuracy": test_acc, 'test_loss': test_loss})
        
    # create plots
    get_plots(run, classes, test_preds, test_labels)
        
    # save fine-tuned model
    torch.save(model, model_save_pth)
        
    wandb.finish()