socat UNIX-LISTEN:jupyter.sock,fork,reuseaddr TCP4:127.0.0.1:8888 & pipenv run jupyter notebook --ip=127.0.0.1 --port=8888
