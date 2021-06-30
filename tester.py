import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import models, datasets, transforms
from collections import OrderedDict

labels = ('Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip')


def load_model():
	model = models.densenet121(pretrained=True)
	model.classifier = nn.Sequential(OrderedDict([
										 ('fc1', nn.Linear(1024, 256)),
										 ('relu', nn.ReLU()),
										 ('fc2', nn.Linear(256, 5)),
										 ('output', nn.LogSoftmax(dim=1))
	]))
	model.load_state_dict(torch.load('checkout.pth'))

	return model


def load_test_dataset():
	test_transform = transforms.Compose([
		transforms.Resize(150),
		transforms.CenterCrop(128),
		transforms.ToTensor()
	])
	testset = datasets.ImageFolder('test/', transform=test_transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
	return testloader



def classify_random_pics():
	testloader = load_test_dataset()
	model = load_model()

	flag = 'go'
	while(flag != 'stop'):
		input, label = next(iter(testloader))
		ps = torch.exp(model(input))
		print(ps)

		fig, ax = plt.subplots()

		y_pos = np.arrange(len(labels))

		ax.barh(y_pos, )