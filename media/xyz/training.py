
import torch
import torch.nn as nn
from torchvision import models

# checking whether GPU is available.
torch.cuda.is_available()

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
device

from my_model import CustomModel

model = CustomModel( 7 )

model = model.to(device) # move to gpu

"""### Loading images"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Resize((128, 128)),

    # transforms.RandomHorizontalFlip(p=0.5),

    # transforms.RandomAffine( degrees=20, translate=(0.2,0.2), shear=10, scale=(0.8, 1.2) ),
    # transforms.ColorJitter( brightness=0.2, contrast=0.2, saturation=0.2 ),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

dataset = datasets.ImageFolder( root='C:/Users/Rey/Desktop/Projects/datasets/raf-db/DATASET/train', transform=transform )

train_loader = DataLoader( dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True )

"""### Training"""

# defining loss function and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=0.001 )

# Training

num_epochs = 50

for i in range( num_epochs ):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    print(f"Epoch [{i+1}/{num_epochs}]" )

    for i, (data, target) in enumerate( train_loader ):

        # clearing old gradiant values
        optimizer.zero_grad()

        # move to GPU
        data, target = data.to( device ), target.to( device )

        # forward propagation
        output = model( data )

        # calculating loss
        loss = loss_fn( output, target )

        # back propagation
        loss.backward()
        optimizer.step()

        # track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max( output.data, 1 )
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Print statistics for the current epoch
        accuracy = 100 * correct / total

        print(f"Batch [{i}] Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%", end='\r' )

    print('\n')

"""### Validataion"""

tform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ds = datasets.ImageFolder( root='C:/Users/Rey/Desktop/Projects/raf-db/test', transform=tform )

val_loader = DataLoader( ds, batch_size=32, num_workers=4, pin_memory=True )

# ----------------------------
# # to split data
# test_size = int( 0.5 * len(ds) )
# val_size = len(ds) - test_size
# test_ds, val_ds = random_split( ds, [test_size, val_size] )
# test_loader = DataLoader( test_ds, batch_size=32, num_workers=4, pin_memory=True )
# ----------------------------

model.eval()

val_loss = 0
val_correct = 0
val_total = 0

with torch.no_grad():
    for val_data, val_target in val_loader:

        val_data, val_target = val_data.to(device), val_target.to(device)

        out = model( val_data )

        val_loss += loss_fn( out, val_target ).item()

        _, val_pred = torch.max( out, 1 )

        val_correct += ( val_pred == val_target ).sum().item()
        val_total += val_target.size(0)

        val_accuracy = val_correct * 100 / val_total

        print(f"val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%", end='\r' )

dataset.classes

# torch.save(model, 'model.pth')
torch.save(model.state_dict(), 'model.pth')

# model = torch.load('model.pth')
# model.load_state_dict(torch.load('model_state_dict.pth'))

from PIL import Image

image = Image.open('C:/Users/Rey/Desktop/Projects/raf-db/pred/ang.jpg')
image = tform(image).unsqueeze(0)

image = image.to(device)

with torch.no_grad():
    out = model( image )

    _, pred = torch.max( out, 1 )


print(pred.item())

dataset.classes