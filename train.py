import torch
import torchvision
from torch import nn
from datasets.dataloader import data_loader
from tqdm import tqdm
import argparse
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader,test_loader,test_data,epochs,model_origin_path,lr,model_save_path):
    criterion = nn.CrossEntropyLoss()
    model = torch.load(model_origin_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    train_accuracies = []  # record accuracy for training
    train_losses = []  # record loss for training
    valid_accuracies = []  # record accuracy for validation
    valid_losses = []  # record loss for validation
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        correct_train = 0   #count number of correct prediction in this batch
        total_train = 0   #count of all data in this batch

        train_bar=tqdm(train_loader,position=0,leave=True) #file=sys.stdout
        for i, data in enumerate(train_bar):
            images, labels = data # get the inputs; data is a list of [images, labels]
            optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            outputs = model(images.to(device)) #output=logit
            loss = criterion(outputs, labels.to(device))
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) #igoring the max of every row but return the max of every column, which is the class
            correct_train += (predicted == labels.to(device)).sum().item()
            total_train += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                    loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)


        model.eval()
        correct_valid = 0
        total_valid = 0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                correct_valid += torch.eq(predict_y, val_labels.to(device)).sum().item()
                total_valid += val_labels.size(0) #=len(testset)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                            epochs)

        valid_accuracy = correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(loss.item())

        val_accurate = acc / len(test_data) #val_num=len(testset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, train_loss, valid_accuracy)) #transteps=len(trainloader)
        # s per iteration, count of iteration = total number of sample / batch size
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(), model_save_path)

    print('Finished Training')

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100, help="batch_size")
    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    train_data,test_data,train_loader,test_loader = data_loader(args.dataset,args.batch_size)
    train(train_loader,test_loader,test_data,config["epochs"],config["model_origin_path"],config["lr"],config["model_save_path"])
