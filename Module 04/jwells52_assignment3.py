import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Global variables
RESULT_INDICES = [
    236,
    478,
    2418,
    2644,
    3968,
    4929,
    5856,
    6437,
    8408,
    9351
] # Indices where there is a big change in tokens when lemmatization or stemming is applied

def write_results(
          _sentences,
          _tokens,
          _stemmed_tokens,
          _lemma_tokens):
    '''Function for writing results of tokenization, stemming, and lemmatization to .txt file'''
    results_file = '/assignment_code/output/results.txt'

    print(f'Writing results to {results_file}...')
    with open(results_file, "w", encoding="utf8") as f:
        for i in RESULT_INDICES:
            print(f'Index: {i}', file=f)
            print('UNPROCESSED STRING', file=f)
            print(_sentences[i], file=f)
            print(file=f)


            print('TOKENIZATION RESULTS', file=f)
            print(_tokens[i], file=f)
            print(file=f)

            print('STEMMING RESULTS', file=f)
            print(_stemmed_tokens[i], file=f)
            print(file=f)

            print('LEMMATIZATION RESULTS', file=f)
            print(_lemma_tokens[i], file=f)
            print('='*100, file=f)
            print(file=f)

if __name__ == "__main__":
    print('Loading data...')
    df = pd.read_csv('Musical_instruments_reviews.csv')

    print('Extract summary column...')
    sentences = df['summary']
    sentences = sentences.apply(lambda str: str.lower())

    print('Tokenizing sentences')
    tokens = sentences.apply(word_tokenize)

    print('Stemming tokens...')
    ps = PorterStemmer()
    stemmed_tokens = tokens.apply(lambda x: [ps.stem(token) for token in x])

    print('Lemmatizing tokens')
    lem = WordNetLemmatizer()
    lemma_tokens = tokens.apply(lambda x: [lem.lemmatize(token) for token in x])

    write_results(sentences, tokens, stemmed_tokens, lemma_tokens)
