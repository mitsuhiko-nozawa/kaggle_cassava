# Cassava
```
kaggle datasets create -p . -r tar
kaggle datasets version -p . -m "Updated data" -r tar
nohup sh -c 'while ps -p {pid} > /dev/null; do sleep 60; done; sh run.sh' &
nohup sh -c 'while ps -p {pid} > /dev/null; do sleep 60; done; sh upload.sh' &
```