# Torch CLF

Pipline classification on the pytorch. Used imagenette https://github.com/fastai/imagenette data with 5% noise as an example

`pip install -r requirements.txt`


## Start
-d: your workdir directory

`python /content/drive/MyDrive/test_task/load_data.py  -d /content/drive/MyDrive/test_task`

-d: your workdir directory

-n: n epoch

-b: batch size

-fn: result filename


`python /content/drive/MyDrive/test_task/train.py -d /content/drive/MyDrive/test_task -e 2 -b 32 -fn 'result'`
