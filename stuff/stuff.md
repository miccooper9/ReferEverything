# Infer


Please download the VSPW stuff annotations from [link](https://github.com/VSPW-dataset/VSPW-dataset-download)

## Modelscope

```bash
bash infer_vspw_stuff_MS.sh #Change the arguments in the script accordingly.
```




## Wan2.1

```bash
bash infer_vspw_stuff_wan.sh #Change the arguments in the script accordingly.
```


# Eval

```bash

python3 eval_stuff.py --result_dir < path to predicted masks >

```