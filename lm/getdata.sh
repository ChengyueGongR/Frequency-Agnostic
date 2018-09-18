echo "=== Acquiring datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
mkdir -p penn
cd penn
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz
mv simple-examples/data/ptb.train.txt train.txt
mv simple-examples/data/ptb.test.txt test.txt
mv simple-examples/data/ptb.valid.txt valid.txt
rm -rf simple-examples.tgz
rm -rf simple-examples/
cd ..

echo "- Downloading text8"
mkdir -p text8
cd text8
wget --quiet --continue http://mattmahoney.net/dc/text8.zip
tar -xzf text8.zip
mv text8 train.txt
#mv test.txt test.txt
#mv valid.txt valid.txt
cd ..

echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt

echo "---"
echo "Happy language modeling :)"
