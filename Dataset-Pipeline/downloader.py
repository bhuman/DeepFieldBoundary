import os
import requests
import sys

from bs4 import BeautifulSoup
from tqdm import tqdm

base_link = 'https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary'


def download(filepath, url):
    response = requests.get(url, stream=True)
    size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with tqdm(total=size, desc=os.path.basename(filepath), unit='iB', file=sys.stdout, unit_scale=True) as progress:
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress.update(len(data))


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'datasets', 'Locations'), exist_ok=True)

    download(os.path.join(os.path.dirname(__file__), 'datasets', 'readme.txt'), f'{base_link}/readme.txt')
    download(os.path.join(os.path.dirname(__file__), 'datasets', 'labels-as-csv.zip'), f'{base_link}/labels-as-csv.zip')
    download(os.path.join(os.path.dirname(__file__), 'datasets', 'fieldboundary.hdf5'), f'{base_link}/fieldboundary.hdf5')

    soup = BeautifulSoup(requests.get(f'{base_link}/Locations').text, features='html.parser')
    for link in soup.select("a[href$='.hdf5']"):
        link_ = link.get('href')
        if not os.path.isfile(os.path.join(os.path.dirname(__file__), 'datasets', 'Locations', link_)):
            download(os.path.join(os.path.dirname(__file__), 'datasets', 'Locations', link_), f'{base_link}/Locations/{link_}')
