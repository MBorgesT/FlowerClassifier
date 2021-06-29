import os
import shutil
import random
from os import walk


TRAIN_SIZE = 0.9


def create_new_folder(dirname):
	if os.path.exists(dirname):
		shutil.rmtree(dirname)
	os.mkdir(dirname)


create_new_folder('train/')
create_new_folder('test/')

for (dirpath, dirnames, filenames) in walk('dataset/'):
	if len(filenames) > 0:
		random.shuffle(filenames)

		split_index = int(len(filenames) * TRAIN_SIZE)
		train = filenames[:split_index]
		test = filenames[split_index:]

		classname = dirpath.split('/')[-1]

		create_new_folder('train/' + classname + '/')
		for file in train:
			shutil.copy(dirpath + '/' + file, 'train/' + classname + '/')

		create_new_folder('test/' + classname)
		for file in test:
			shutil.copy(dirpath + '/' + file, 'test/' + classname + '/')
