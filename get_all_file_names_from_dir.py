'''
Get all the names of files in a folder to a text file.
Useful for training purposes.

'''
import os

# Use json folder(labels ) to get the list from directory

file_list_imgs = ' ' # Output text file
labels_folder = ' ' # Input labels folder


f = open(file_list_imgs, 'a')

count = 0
labels_dir = sorted([(file[:-5]) for file in os.listdir(labels_folder) if 'json' in file])


for file in labels_dir:
	fname = str(file)
	print(fname)
	count = count + 1
	f.write(fname+'\n')

print(count)
f.close()