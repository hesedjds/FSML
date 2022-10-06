import json, os 

cwd = os.getcwd()
data_path = os.path.join(cwd, 'img')
split_path = os.path.join(cwd, 'split')

for file_name in os.listdir(split_path):
    if file_name.split('.')[-1] != 'json': continue 
    with open(os.path.join(split_path, file_name)) as f:
        data = json.load(f)
    image_names = []
    for name in data['image_names']:
        image_names.append(os.path.join(data_path, *name.split('/')[-2:]))
    data['image_names'] = image_names
    with open(file_name, 'w') as f:
        json.dump(data, f)