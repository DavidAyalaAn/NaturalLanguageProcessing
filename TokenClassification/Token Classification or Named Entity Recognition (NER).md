

# Finetuning
This is a basic flow for finetuning applied to token classification.
<img src="TokenClassification_01.png" width="750px" />
# Label Mapping
We need to define a numerical dictionary to classify our tokens.
Depending on the problem we would need to choose a good labeling criteria.

```python
#Dictionary for mapping a numerical id to labels
id2label = {
    0: "O",        # Outside of a term/definition
    1: "B-TERM",   # Beginning of a term
    2: "I-TERM",   # Inside a term
    3: "B-DEF",    # Beginning of a definition
    4: "I-DEF",    # Inside a definition
}

#Reverse dictionary to map labels to numerical ids
label2id = {label: id for id, label in id2label.items()}

#store the number of labels we are considering
num_labels = len(id2label)
```



## Tokenize
In this step we transform the text sample into tokens which are going to be switch by its numerical representation.

First, we load the appropriate tokenizer for the model we are gonna use.
```python
from transformers import AutoTokenizer #Used to load an appropriated tokenizer

#Loads the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

We have different tokenizer options:
- bert-base-cased


Then we will need to transform our text in tokens, the following applies only for one text but we are suppose to applied to a full training set.
```python
full_text = 'Luke Jedi' #this corresponds to our text sample

#Transform the text into tokens
tokenized_inputs = tokenizer(
        full_text, #corresponds to our text sample
        truncation=True, #Allows the text truncation to limit la length
        max_length=512, #Truncate the text to 512 characters
        return_offsets_mapping=True
    )


```

# Aling Labels

In this step, we assign a label to each character token.
This will stablish the pattern used for the training.

<img src="TokenClassification_02.png" width="250px" />

```python
#First we set the lengt
labels = [-100] * len(tokenized_inputs['input_ids'])  # -100 is used to ignore tokens in loss calculation
    
    # Find the start and end character positions of the term and definition in the full text
    term_start_char = full_text.find(term_text)
    term_end_char = term_start_char + len(term_text)
    def_start_char = full_text.find(def_text, term_end_char)
    def_end_char = def_start_char + len(def_text)
    
    for i,offset in enumerate(tokenized_inputs['offset_mapping']):
        #Retrieve the start and end character positions of the token respective to the full text
        token_start_char, token_end_char = offset
        
        if token_start_char == token_end_char and token_start_char == 0 and i > 0:
            # If the token is empty (e.g., a space or punctuation), skip it
            continue
        
        #Compare the token's character positions with the term and definition positions to assign corresponding labels
        if  term_start_char <= token_start_char < term_end_char:
            if token_start_char == term_start_char:
                labels[i] = label2id['B-TERM']
            else:
                labels[i] = label2id['I-TERM']
        elif def_start_char <= token_start_char < def_end_char:
            if token_start_char == def_start_char:
                labels[i] = label2id['B-DEF']
            else:
                labels[i] = label2id['I-DEF']
        else:
            labels[i] = label2id['O']
    
    tokenized_inputs['labels'] = labels
    tokenized_inputs.pop('offset_mapping')  # Remove offset_mapping as it's not needed for training
    tokenized_inputs.pop('token_type_ids')  # Remove offset_mapping as it's not needed for training

```
# Data Collation
This part just divide all our samples in batches, so it is not all sent to the model at once.
```python
data_collator = DataCollatorForTokenClassification(tokenizer)
```
# Model Definition
In this part, define the training parameters for the fine tunning.
```python
model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-cased', 
    num_labels= num_labels,
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir='./term_def_model',
    eval_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_steps=10,
    load_best_model_at_end=True,
    #push_to_hub=False,
    disable_tqdm=False,
    fp16=True,
    no_cuda=False # Set to True if you don't have a GPU
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = term_def_dataset['train'],
    eval_dataset = term_def_dataset['val'],
    #processing_class = tokenizer,
    data_collator=data_collator,
    callbacks=[PrintCallback()]
)

```

To train the model we use:
```python
trainer.train()
```

# Model Testing

By last we test our trained model with unseen data.

```python
from transformers import pipeline
best_ckpt = trainer.state.best_model_checkpoint
print('Best checkpoint: {}'.format(best_ckpt))

if best_ckpt:
    model = AutoModelForTokenClassification.from_pretrained(best_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt)

pipe = pipeline(
    'token-classification',
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 #0 for GPU and -1 for CPU
)
```


```python

raw_texts = list(test_df['text'])
predictions = pipe(raw_texts)

records = []

for i, (text,pred) in enumerate(zip(raw_texts,predictions)):
    terms_entities = [p for p in pred if p['entity_group'] == 'TERM']
    defs_entities = [p for p in pred if p['entity_group'] == 'DEF']
    
    extrated_term = ""
    extrated_def = ""
    
    if terms_entities:
        extrated_term = terms_entities[0]['word']
    
    if defs_entities:
        extrated_def = defs_entities[0]['word']
        
    records.append({
        'original_text': text,
        'predicted_term': extrated_term,
        'predicted_definition': extrated_def
    })
```