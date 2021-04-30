import wget
import tarfile
import os
import argparse

parser = argparse.ArgumentParser(description='Load data')
parser.add_argument("-d", "--to_dir", required=True, help = 'Path to workdir')
args = parser.parse_args()

url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz'
path = args.to_dir
dir_for_data = 'data'


def get_data(url, path, dir_for_data):
    """
    Function for load data
    :param url: hard param = link to resource with data
    :param path: path to workdir
    :param dir_for_data: hard param = 'data'
    :return: created dir with unzip data
    """
    all_path = os.path.join(path, dir_for_data)
    try:
        os.mkdir(all_path)
        wget.download(url, all_path)
    except:
        pass
    my_tar = tarfile.open(os.path.join(all_path, 'imagewoof2-320.tgz'))
    my_tar.extractall(all_path)  # specify which folder to extract to
    my_tar.close()


if __name__ == '__main__':
    get_data(url, path, dir_for_data)