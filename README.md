## Description 
GPT-2 benchmarking on wikitext-103. 
Derived from  https://gist.github.com/thomwolf/ca135416a30ea387aa20edaa9b21f0ed.

## Environment set up
Make a new virtual environment. From your home directory run:
```
python3 -m venv env && source env/bin/activate
```

Install pytorch for gpu:
```
pip3 install torch==1.13.0+cu116 -f 'https://download.pytorch.org/whl/torch_stable.html'
```

Install rest of requirements:
```
pip3 install -r gpt-2/requirements.txt
```

## Download data
To download and extract data into `gpt-2` dir run:
```
cd ~/gpt-2 && bash download_data.sh
```

## Train GPT-2 on wikitext-103 with word level tokenizer
From `~/gpt-2` run:
```
python gpt2-wikitext-103-word-level.py
```
