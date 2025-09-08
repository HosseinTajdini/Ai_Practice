"""
Train a Pytorch  image classification model using device-agnostic code.
"""
from typing import Dict, List,Tuple
import torch
from  torchvision import transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer
import data_setup , engine , utils , model_builder
from torch import nn
# Setup hyperparameters
NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001



train_dir = "data/pizza_steak_sushi/train"
test_dir ="data/pizza_steak_sushi/test"

device='cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])







start_time=timer()
if __name__ == "__main__":

    train_dataloader,test_dataloader,class_names=data_setup.create_Dataloaders(train_dir,test_dir,data_transform,BATCH_SIZE,0)
    model_1=model_builder.TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS,output_shape=len(class_names))
    optimizer=torch.optim.SGD(lr=LEARNING_RATE,params=model_1.parameters())
    loss_fn = nn.CrossEntropyLoss()
    print(class_names[0])
    result=engine.train(model=model_1,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device)
    utils.save_model(model=model_1,
                 target_dir="my_models",
                 model_name="05_section_model.pth")

end_time=timer()
print(f"total trainig time is {end_time-start_time}")
#save model
