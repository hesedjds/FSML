#!/usr/bin/env bash
#download dataset
#https://drive.google.com/file/d/191cFzwwNTzG_mHUDABF0Nh77cI6pa-qq/view?usp=sharing
#https://drive.google.com/file/d/0B7EVK8r0v71pa2EyNEJ0dE9zbU0/view?usp=sharing&resourcekey=0-CPiKS-AiE8IDonk54WJ5_w

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0" -O fashion.zip && rm -rf /tmp/cookies.txt

unzip fashion.zip 
rm -rf fashion.zip 

python write_fashion_dataset.py 