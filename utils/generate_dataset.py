from os.path import join
import yaml, pickle

def __readFromFiles(l_fileNames):
    res = []
    for fName in l_fileNames:
        with open(fName, 'r', encoding='utf-8') as f:
            content = f.read()
        res.append(content)
    return res

def __generate_dataset(recipe, path):
    dataset = []
    for key in recipe.keys():
        l_TjFreeFNames = None
        l_TjInFNames = None
        if 'TjFree' in recipe[key].keys():
            l_TjFreeFNames = [join(path, key, fName) for fName in recipe[key]['TjFree']]
        if 'TjIn' in recipe[key].keys():
            l_TjInFNames = [join(path, key, fName) for fName in recipe[key]['TjIn']]
        if l_TjFreeFNames != None:
            TjFreeSample = __readFromFiles(l_TjFreeFNames)
            dataset.append({'name':key, 'rtl':TjFreeSample, 'label':'TjFree'})
        if l_TjInFNames != None:
            TjInSample = __readFromFiles(l_TjInFNames)
            dataset.append({'name':key, 'rtl':TjInSample, 'label':'TjIn'})
    return dataset

if __name__ == '__main__':
    with open('recipe.yaml', 'r', encoding='utf-8') as f:
        recipe = f.read()
    recipe = yaml.load(recipe, Loader=yaml.CLoader)
    dataset = __generate_dataset(recipe, './datasets')
    with open('datasets/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    # with open('datasets/dataset.pkl', 'rb') as f:
    #     print(pickle.load(f))