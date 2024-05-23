import os

def rename_images(directory):
    i = 0  # Initialize the variable i
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            i += 1
            new_name = str(i) + '.jpg'
            new_path = os.path.join(directory, new_name)
            if not os.path.exists(new_path):  # Check if the new file name already exists
                os.rename(os.path.join(directory, filename), new_path)

    return print(i)
# Specify the directory where the images are located

for i in range(27):

    directory = f'./data/{i}/'
    rename_images(directory)
