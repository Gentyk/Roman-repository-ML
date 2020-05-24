import os

import pandas

from lab5_4 import image_features2


def run(images):
    all_features = []
    for j, i in enumerate(images):
        if j % 100 == 0:
            print(j)
        name = i.split("_")[0]
        try:
            features = None
            for por in [30, 0, 60, 90, -10]:
                try:
                    features = image_features2(f"{directory}/{i}", por=por, name=name)
                    break
                except:
                    pass
            if not features:
                print('skip',j,i)
                continue
            all_features.append(features.copy())
        except:
            print(j, i)
            raise

    df = pandas.DataFrame(all_features)
    df.to_csv(r'features_train.csv', index=False, header=False)

directory = './br-coins/classification_dataset/all'
files = os.listdir(directory)
image_paths = [i for i in filter(lambda x: x.endswith('.jpg'), files)]
print(len(image_paths))
run(image_paths)