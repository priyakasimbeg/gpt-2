## Description 
GPT-2 benchmarking on wikitext-103. 
Adapted from  https://gist.github.com/thomwolf/ca135416a30ea387aa20edaa9b21f0ed.

## Environment set up
Make a new virtual environment. From your home directory run:
```
python3 venv env && env/bin/activate
```

Install pytorch for gpu:
```
pip3 install torch==1.13.0+cu117 -f 'https://download.pytorch.org/whl/torch_stable.html'
```

Install rest of requirements:
```
pip3 install -r requirements.txt
```

