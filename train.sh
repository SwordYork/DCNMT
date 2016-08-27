#!/usr/bin/env sh
mose_git=https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts

echo 'select the source language: en, cs, fi, fr, ru, de'
read -p '==> ' source_language
echo "the selected source language is $source_language"


echo 'select the target language: en, cs, fi, fr, ru, de'
read -p '==> ' target_language
echo "the selected target language is $target_language"

if [ $target_language = $source_language ]; then
    echo 'languages should be different'
    exit -1
fi

if [ ! -d share/nonbreaking_prefixes ]; then
    mkdir -p share/nonbreaking_prefixes 
fi

echo "downloading nonbreaking_prefix $source_language ..."
curl -s $mose_git/share/nonbreaking_prefixes/nonbreaking_prefix.$source_language > share/nonbreaking_prefixes/nonbreaking_prefix.$source_language
echo "downloading nonbreaking_prefix $target_language ..."
curl -s $mose_git/share/nonbreaking_prefixes/nonbreaking_prefix.$target_language > share/nonbreaking_prefixes/nonbreaking_prefix.$target_language

if [ ! -d data ]; then
    echo 'creating data directory...'
    mkdir data
fi



cp preprocess/create_vocab.py preprocess/shuffle_data.py data/

echo 'cd to data directory'
cd data
if [ ! -f tokenizer.perl ]; then
    echo 'downloading tokenizer.perl'
    curl -s $mose_git/tokenizer/tokenizer.perl > tokenizer.perl
fi
if [ ! -f multi-bleu.perl ]; then
    echo 'downloading multi-bleu.perl'
    curl -s $mose_git/generic/multi-bleu.perl > multi-bleu.perl
fi

echo 'please download corresponding datasets from WMT15 manually'
echo 'and put the parallel corpus in the data directory'
echo 'then enter the file name of source language dataset'
read -p '==> ' source_data
echo 'the file name of target language dataset'
read -p '==> ' target_data

if [ ! -f $source_data ]; then
    echo 'no such source dataset file'
    exit -1
fi

if [ ! -f $target_data ]; then
    echo 'no such target dataset file'
    exit -1
fi

tok_source_file=all.$source_language-$target_language.$source_language.tok
tok_target_file=all.$source_language-$target_language.$target_language.tok

perl tokenizer.perl -l $source_language -threads 4 -no-escape < $source_data > $tok_source_file
perl tokenizer.perl -l $target_language -threads 4 -no-escape < $target_data > $tok_target_file

echo 'please put the validation and test dataset in the data directory'
echo 'the file name of source validation set:'
read -p '==>' src_val
echo 'the file name of target validation set:'
read -p '==>' trg_val
echo 'the file name of source test set:'
read -p '==>' src_test
echo 'the file name of targe test set:'
read -p '==>' trg_test

if [ ! -f $src_val ]; then
    echo 'no such source validation file'
    exit -1
fi

if [ ! -f $trg_val ]; then
    echo 'no such target validation file'
    exit -1
fi

if [ ! -f $src_test ]; then
    echo 'no such source test file'
    exit -1
fi

if [ ! -f $trg_test ]; then
    echo 'no such target test file'
    exit -1
fi

perl tokenizer.perl -l $source_language -threads 4 -no-escape < $src_val > $src_val.tok
perl tokenizer.perl -l $source_language -threads 4 -no-escape < $src_test > $src_test.tok
perl tokenizer.perl -l $target_language -threads 4 -no-escape < $trg_val > $trg_val.tok
perl tokenizer.perl -l $target_language -threads 4 -no-escape < $trg_test > $trg_test.tok


echo 'please ensure there is enough disk space'
python shuffle_data.py $tok_source_file $tok_target_file

echo 'please enter the size of source language vocabulary (120 is enough):'
read -p '==>' src_vocab_size
echo 'please enter the size of target language vocabulary (120 is enough):'
read -p '==>' trg_vocab_size
python create_vocab.py $source_language $target_language $src_vocab_size $trg_vocab_size $tok_source_file.shuf $tok_target_file.shuf

cd ..
if [ -f configurations.py ]; then
    cp configurations.py configurations_backup.py
fi 

cp configurations_template.py configurations.py

sed -i "s/--src_lang--/$source_language/" configurations.py
sed -i "s/--trg_lang--/$target_language/" configurations.py
sed -i "s/--src_vocab_size--/$src_vocab_size/" configurations.py
sed -i "s/--trg_vocab_size--/$trg_vocab_size/" configurations.py
sed -i "s/--src_val--/$src_val/" configurations.py
sed -i "s/--trg_val--/$trg_val/" configurations.py
sed -i "s/--src_test--/$src_test/" configurations.py
sed -i "s/--trg_test--/$trg_test/" configurations.py

echo "ok! just run 'python training.py'"




