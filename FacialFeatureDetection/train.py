from ensurepip import version
from model import ResNetAttributes,FocalLoss
from dataset import CelebADataset, translate,unnormalize,transform_image
import os
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
import cv2
# Learning rate scheduler
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(
        epochs:int,
        batch_size:int,
        save_path:str="models/",
        model_path:str="models/",
        pretrained:bool=False,
        model_name:str=None,
        learning_rate:float=0.001,
        version:str="resnet18"
        ):

    """
    Train the model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load the dataset
    dataset = CelebADataset(label_path="data/list_attr_celeba.csv",
                            img_dir="data/img_align_celeba/img_align_celeba/",
                            train_ratio=0.85,
                            test_ratio=0.05,
                            val_ratio=0.10,
                            batch_size=batch_size,
                            shuffle=True)
    dataloader = dataset.get_dataloader()

    # Load the model
    model = ResNetAttributes(pretrained=pretrained,version=version)
    print(model)
    model.to(device)
    if not os.path.isdir("results/"+model.name):
        os.mkdir("results/"+model.name)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    #criterion = FocalLoss(gamma=2, alpha=None, size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_sched = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,steps_per_epoch=len(dataloader["train"]), epochs=epochs)
    prev_accuracy = 0
    ave_loss_history  = []
    # Train the model
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} training {model.name}")
        for i, (images, labels) in enumerate(tqdm(dataloader["train"])):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            # print(f"Shape of outputs: {outputs.shape}")
            # print(f"Shape of labels: {labels.shape}")
            #loss = sum([criterion(outputs[:,j], labels[:,j]) for j in range(labels.shape[1])])
            loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels)
            ave_loss_history.append(loss.cpu().detach())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 250 == 0:
                print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                        .format(epoch+1, epochs, i+1, len(dataloader["train"]), loss.detach()))
                accuracy = eval_model(model,dataloader,threshold=0.5)
                plt.plot(ave_loss_history)
                plt.savefig(f"results/{model.name}/loss_history.png")
            lr_sched.step()
            curr_lr = lr_sched.get_last_lr()
        print("Learning rate: {}".format(curr_lr))
        print("Evaluating model")
        plt.plot(ave_loss_history)
        plt.savefig(f"results/{model.name}/loss_history.png")
        accuracy = eval_model(model,dataloader,threshold=0.5)
        if accuracy > prev_accuracy:
            model.save(best=True)
            prev_accuracy = accuracy
        else:
            model.save(best=False,epoch=epoch)

    # Save the model checkpoint
    model.save()
    print("Model saved to {}".format(save_path+model.name+".pth"))
    return model
def eval_model(model,dataloader,threshold=0.0,mode:str="val"):
    """
    Evaluate the model
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader[mode]):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            outputs[outputs<threshold] = 0
            outputs[outputs>=threshold] = 1
            predicted = outputs
            total += labels.size(0)*40
            correct += (predicted == labels).sum().item()
            #print(f"Number of correct predictions: {correct}")
    print('Accuracy of the network on the images: %d %%' % (100 * correct / total), mode)
    model.train()
    return correct/total
def create_model(model_path:str,pretrained:bool=False,version="resnet18"):
    """
    Create a model
    """
    model = ResNetAttributes(model_path = model_path,pretrained=pretrained,version=version)
    return model
def display_predictions(model,dataloader,threshold,mode:str="test"):
    """
    Evaluate the model
    """
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader[mode]):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            predicted = outputs
            predicted[outputs<threshold] = 0
            predicted[outputs>=threshold] = 1
            attributes = translate(labels[0])
            pred_attrr = translate(predicted[0])
            print(f"\nAttributes: {attributes}")
            print(f"Predicted attributes: {pred_attrr}")
            plt.imshow(unnormalize(images[0].cpu()))
            plt.show()
            break

def single_inference(model,image_path,device,threshold):
    """
    Evaluate the model
    """
    model.eval()
    image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    image = transform_image(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        predicted = outputs
        predicted[outputs<threshold] = 0
        predicted[outputs>=threshold] = 1
        pred_attrr = translate(predicted[0])
        print(f"Predicted attributes: {pred_attrr}")
        plt.imshow(unnormalize(image[0].cpu()))
        plt.show()
if __name__ == "__main__":
    train(epochs=3,save_path="models/",model_path="models/",batch_size=256,pretrained=False,learning_rate=0.002,version="resnet34")
    # model = create_model(model_path="models/coolibah_licitation.pth",pretrained=False,version="resnet34")
    # dataset = CelebADataset(label_path="data/list_attr_celeba.csv",
    #                         img_dir="data/img_align_celeba/img_align_celeba/",
    #                         train_ratio=0.85,
    #                         test_ratio=0.05,
    #                         val_ratio=0.10,
    #                         batch_size=64,
    #                         shuffle=True)
    # dataloader = dataset.get_dataloader()
    # eval_model(model,dataloader=dataloader,threshold=0.5,mode="test")
    #display_predictions(model,dataloader,model.device,threshold=0.5,mode="test")