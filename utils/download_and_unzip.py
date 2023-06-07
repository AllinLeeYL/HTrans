import urllib.request as ulb
from bs4 import BeautifulSoup
from os.path import exists, join, isdir, basename
import os, re, pickle, json

def getWeb(url):
    request = ulb.Request(url)
    request.add_header('user-agent', 'Mozilla/5.0')
    response = ulb.urlopen(request)
    return response.read().decode('utf-8'), response.getcode()
    # with open('utils/content.html', 'r') as f:
    #     content = f.read()
    # return content, 200

def __transform(jsonStr):
    l_benchmarks = json.loads(jsonStr)
    for i, benc in enumerate(l_benchmarks):
        l_benchmarks[i]['links'] = ['https://www.trust-hub.org' + link for link in benc['links']]
    return l_benchmarks

def __filterHTML(content): # abandoned
    soup = BeautifulSoup(content, 'html.parser')
    l_benchmarks = soup.find_all('app-benchmark-item', class_='ng-star-inserted')
    for i, benc in enumerate(l_benchmarks):
        l_urls = benc.find_all(class_='trojan-bench-file')
        name = benc.find('a', attrs={'role':'button'}).text.strip(' ')
        for j, res in enumerate(l_urls):
            l_urls[j] = 'https://www.trust-hub.org' + res.find('a').attrs['href']
        l_benchmarks[i] = {'name':name, 'urls':l_urls}
    return l_benchmarks

def __download_and_unzip(l_benchmarks, path):
    l_err = []
    l_multiFile = []
    if not exists(path):
        os.mkdir(path)
    for benc in l_benchmarks:
        bencName = benc['name']
        bencPath = join(path, bencName)
        if not exists(bencPath):
            os.mkdir(bencPath)
        if len(benc['links']) != 1:
            l_multiFile.append(bencName)
        for url in benc['links']: # download file
            fName = basename(url)
            os.system('wget -P ' + bencPath + ' ' + url)
            # unzip file
            if re.match(r'.*\.zip', fName):
                os.system('unzip -q -n -d ' + path + ' ' + join(bencPath, fName))
            elif re.match(r'.*\.rar', fName):
                os.system('unrar x -y ' + join(bencPath, fName) + ' ' + path)
            else:
                l_err.append(url)
    print('error:', l_err)
    print('multiFile', l_multiFile)

def __save2file(obj, path):
    if re.match(r'.*\.json', path):
        with open(path, 'w') as f:
            json.dump(obj, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

if __name__ == '__main__':
    url = 'https://www.trust-hub.org/assets/json/benchmarks-trojan-chip.json'
    path = './datasets'
    content, code = getWeb(url)
    if code == 200:
        l_benchmarks = __transform(content)
        __download_and_unzip(l_benchmarks, path)
        # samples, labels = __generate_dataset(recipe, path)
        # __save2file({'sample':samples, 'label':labels}, './datasets/dataset.json')
