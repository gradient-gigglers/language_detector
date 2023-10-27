## Theera


```sh
$ ssh -oIdentitiesOnly=yes -oIdentityFile=None -i ~/.ssh/k3s-agent-7 root@65.109.100.94
# install miniconda and all the packages needed, follow:
# https://github.com/besarthoxhaj/tiny-team-example/tree/master
# https://github.com/besarthoxhaj/lora
$ conda install pip
$ pip install fastapi "uvicorn[standard]"
$ pip install pandas pyarrow
$ pip install sentencepiece torch
$ uvicorn server:app --host 0.0.0.0 --port 8080 --reload
# 65.109.100.94:8080
# Let's test the server
$ curl -X POST -H "Content-Type: application/json" -d '{"text":"maybe"}' http://localhost:8080/what_language_is_this
```


```sh
$ wget https://huggingface.co/datasets/iix/Parquet_FIles/resolve/main/Flores7Lang.parquet -O Flores7Lang.parquet
```