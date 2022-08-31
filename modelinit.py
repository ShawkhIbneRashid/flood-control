from buildmodel import Build_Model
from dataloader import Data_Loader
import torch
import torch.nn as nn
import torch.optim as optim
# Initialize the model for this run
if __name__ == '__main__': 
    #model_name = "inception"
    model_name = "inception" #change to inception for inception v3 training
    num_classes = 3
    batch_size = 8
    num_epochs = 15
    feature_extract = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    build_model = Build_Model(device)
    model_ft, input_size = build_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    
    print(model_ft)
    
    model_ft = model_ft.to(device)
    
    
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    data_loader = Data_Loader(299, batch_size)#use 224 for densenet and 299 for inception v3
    dataloaders_dict = data_loader.dataform()
    # Train and evaluate

    model_ft, hist = build_model.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    torch.save(model_ft.state_dict(), model_name+"_0.pth")