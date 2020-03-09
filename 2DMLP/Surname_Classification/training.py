import helper as hp
import torch.nn as nn
import torch as t
from argparse import Namespace
import pandas as pd
import surname_dataset as sd
import surname_classifier as sc
import torch.optim as optim 
from tqdm import tqdm 
import matplotlib.pyplot as plt

args = Namespace(
    # Data and path setting
    surname_csv = "data/surnames.csv",
    surname_processed_csv = "data/surnames_with_splits",
    vectorizer_file = "vectorizer.json",
    model_file = "model.pth",
    save_dir = "surname_mlp",

    # model hyper parameters
    hidden_size = 3,

    # training hyper parameters
    seed = 24,
    batch_size = 24,
    num_epochs = 10,
    early_stopping_criteria = 5,
    learning_rate = 0.001,

    # runtime 
    cuda = False,
    reload_from_file = False
)

# check if the cuda is available
if t.cuda.is_available():
    args.cuda = True

hp.set_seed_everywhere(args.seed)

if args.reload_from_file:
    # instantiate surname_dataset from an existing vectorizer and load the dataset
    print("Reloading starts")
    # dataset is the instantiation of the surname_dataset class
    dataset = sd.load_dataset_and_load_vectorizer(args.surname_processed_csv,args.vectorizer_file)
    print("Reloading ends")
else:
    # instantiate surname_dataset and make the vectorizer from the DataFrames
    print("Dataset Creating Starts")
    dataset = sd.SurnameDataset.load_dataset_and_make_vectorizer(args.surname_processed_csv)
    print("Dataset Creating Ends")

vectorizer = dataset.get_vectorizer()
# print(vectorizer.get_vector_size())
classifier = sc.SurnameClassifier(in_size = vectorizer.get_vector_size(),\
    hidden_size = args.hidden_size, out_size = len(vectorizer.nation_vocab) )
print(classifier)


# Define the loss funcation and optimizer  
loss_func = nn.CrossEntropyLoss(weight = dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr = args.learning_rate)

# define the scheduler to tweak the learning rate
# ReduceLRonPlateau with model "min" means to reduce the learning rate by factor when the evaluating indicator starts to decrease. 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.5, patience=  1)


train_state = hp.make_train_state(args)

# To visualize the progress
print(args.num_epochs)

try:
    for epoch in tqdm(range(args.num_epochs),desc = "current epoch", leave = True):
        train_state['epoch_index'] = epoch

        dataset.set_split("train")

        batch_generator = sd.generate_batches(dataset,batch_size = args.batch_size)

        running_loss = 0.0
        running_acc = 0.0
        
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):

            # zero the gradient 
            optimizer.zero_grad()

            # compute the output
            y_pred = classifier(batch_dict['surname'])
            
            # compute the loss
            loss = loss_func(y_pred, batch_dict['nation'])
            loss_value = loss.item()
            # compute the moving average loss for plotting
            running_loss = (loss_value - running_loss) / (batch_index + 1)
            
            # propagate the gradient
            loss.backward()

            # update the parameters 
            optimizer.step()

            #compute the accuracy and moving average accuracy 
            acc_value = hp.compute_accuracy(y_pred, batch_dict['nation'])
            running_acc = (acc_value - running_acc) / (batch_index + 1)
        
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        # Iterate over val dataset

        dataset.set_split("val")
        batch_generator = sd.generate_batches(dataset, args.batch_size)

        running_loss = 0.0
        running_acc = 0.0
        classifier.eval()

        for batch_index,batch_dict in enumerate(batch_generator):
            
            # compute the output 
            y_pred = classifier(batch_dict['surname'])

            # compute the loss
            loss = loss_func(y_pred, batch_dict['nation'])
            loss_value = loss.item()
            # compute the Moving Average.
            running_loss = (loss_value - running_loss) / (batch_index + 1)

            # compute the accuracy 
            acc = hp.compute_accuracy(y_pred, batch_dict['nation'])
            running_acc = (acc - running_acc) / (batch_index + 1)

        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        train_state = hp.update_train_state(args = args, model = classifier, train_state = train_state)

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            break

    plt.plot(train_state['train_loss'])
    plt.show()

except KeyboardInterrupt:
    print("Existing Loop")

# do the inference

# load the parameters from the best model
classifier = t.load(args.model_file)

running_loss = 0.0
running_acc = 0.0

dataset.set_split("test")

batch_generator = sd.generate_batches(dataset,args.batch_size)

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the loss