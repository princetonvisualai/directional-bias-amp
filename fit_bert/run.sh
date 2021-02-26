# set up the information for when base correlation source is wikpedia
python3 read_wiki2.py

# s represents the base source version
for s in 0 1 2
do 
    python3 fit_sentences.py --base_source $s --pronouns 2 --version 10
done

python3 fit_sentences.py --base_source 0 --pronouns 3 --version 10

python3 plot.py
