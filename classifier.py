import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def load_dataset():
	train_transform = transforms.Compose([
		transforms.RandomRotation(30),
		transforms.RandomResizedCrop(128),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_transform = transforms.Compose([
		transforms.Resize(150),
		transforms.CenterCrop(128),
		transforms.ToTensor()
	])

	trainset = datasets.ImageFolder('train/', transform=train_transform)
	testset = datasets.ImageFolder('test/', transform=test_transform)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

	return trainloader, testloader


if __name__ == '__main__':
	trainloader, testloader = load_dataset()

	'''
	print(torch.cuda.is_available())
	print(torch.version.hip)
	input('Go?')
	'''

	model = models.densenet121(pretrained=True)
	model.classifier = nn.Sequential(OrderedDict([
										 ('fc1', nn.Linear(1024, 256)),
										 ('relu', nn.ReLU()),
										 ('fc2', nn.Linear(256, 5)),
										 ('output', nn.LogSoftmax(dim=1))
	]))
	model.cuda()

	device = torch.device('cuda')

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

	epochs = 10
	steps = 0
	running_loss = 0
	print_every = 24
	print('Running...')
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1

			#inputs, labels = inputs.to(device), labels.to(device)
			inputs, labels = inputs.cuda(), labels.cuda()

			logps = model(inputs)
			loss = criterion(logps, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if steps % print_every == 0:
				test_loss = 0
				accuracy = 0
				model.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.cuda(), labels.cuda()
						
						logps = model(inputs)
						batch_loss = criterion(logps, labels)

						test_loss += batch_loss.item()

						# Calculate accuracy
						ps = torch.exp(logps) 
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

				print(f"Epoch {epoch+1}/{epochs}.. "
					f"Train loss: {running_loss/print_every:.3f}.. "
					f"Test loss: {test_loss/len(testloader):.3f}.. "
					f"Test accuracy: {accuracy/len(testloader):.3f}")
				running_loss = 0
				model.train()
