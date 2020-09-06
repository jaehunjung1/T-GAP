mkdir icews14
wget --output-document=icews14/train.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews14/icews_2014_train.txt
wget --output-document=icews14/valid.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews14/icews_2014_valid.txt
wget --output-document=icews14/test.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews14/icews_2014_test.txt

mkdir icews05-15
wget --output-document=icews05-15/train.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews05-15/icews_2005-2015_train.txt
wget --output-document=icews05-15/valid.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews05-15/icews_2005-2015_valid.txt
wget --output-document=icews05-15/test.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/icews05-15/icews_2005-2015_testtxt

mkdir wikidata11k
wget --output-document=wikidata11k/train.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/wikidata/wiki_train.txt
wget --output-document=wikidata11k/valid.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/wikidata/wiki_valid.txt
wget --output-document=wikidata11k/test.txt https://raw.githubusercontent.com/nle-ml/mmkb/master/TemporalKGs/wikidata/wiki_test.txt

