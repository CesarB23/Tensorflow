import torch
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
print(torch.cuda.is_available())