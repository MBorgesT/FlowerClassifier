import torch
import matplotlib.pyplot as plt
import time
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def load_dataset():
	train_transform = transforms.Compose([
		transforms.RandomRotation(30),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
    	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	test_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor()
	])

	trainset = datasets.ImageFolder('train/', transform=train_transform)
	testset = datasets.ImageFolder('test/', transform=test_transform)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

	return trainloader, testloader


def plot_loss_graph(train, test):
	plt.plot(train, label='Training loss')
	plt.plot(test, label='Validation loss')
	plt.legend(frameon=False)
	plt.savefig('graphs/losses.png')


if __name__ == '__main__':
	is_cuda = False
	if torch.cuda.is_available():
		is_cuda = True

	trainloader, testloader = load_dataset()

	model = models.densenet121(pretrained=True)
	model.classifier = nn.Sequential(*[
		nn.Linear(1024, 256),
		nn.ReLU(),
		nn.Linear(256, 5),
		nn.LogSoftmax(dim=1)
	])
	if is_cuda:
		model.cuda()

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.0035)

	epochs = 4
	steps = 0
	running_loss = 0
	print_every = 24

	train_losses = []
	validation_losses = []

	last_print = time.time()

	print('Running...')
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1

			if is_cuda:
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
						if is_cuda:
							inputs, labels = inputs.cuda(), labels.cuda()
						
						logps = model(inputs)
						batch_loss = criterion(logps, labels)

						test_loss += batch_loss.item()

						# Calculate accuracy
						ps = torch.exp(logps) 
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

				train_loss = running_loss/print_every
				test_loss = test_loss/len(testloader)

				print(f"Epoch {epoch+1}/{epochs}   "
					f"Train loss: {train_loss:.3f}   "
					f"Test loss: {test_loss:.3f}   "
					f"Test accuracy: {accuracy/len(testloader):.3f}"
					f"      {time.time() - last_print:.1f} sec")
				running_loss = 0
				last_print = time.time()
				model.train()

				train_losses.append(train_loss)
				validation_losses.append(test_loss)
	
	torch.save(model.state_dict(), 'checkout.pth')
	plot_loss_graph(train_loss, validation_losses)
