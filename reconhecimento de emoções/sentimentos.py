import nltk

#nltk.download()
#nltk.download faz o download das palavras do nltk

frases_alegria = [
    ("Estou radiante com a qualidade do produto; superou todas as minhas expectativas!", 'alegria'),
    ("Que experiência incrível! Este serviço não só atendeu, mas superou minhas necessidades.", 'alegria'),
    ("Estou extremamente feliz com a eficácia deste produto. Realmente fez a diferença para mim.", 'alegria'),
    ("A alegria é imensa ao constatar que este serviço não só é eficiente, mas também surpreendentemente fácil de usar.", 'alegria'),
    ("Este produto é um verdadeiro achado! Cumpre tudo o que promete e mais um pouco.", 'alegria'),
    ("A satisfação é palpável ao constatar que o investimento neste serviço foi mais do que válido.", 'alegria'),
    ("Não consigo conter minha alegria ao recomendar este produto. Simplesmente incrível!", 'alegria'),
    ("A qualidade excepcional deste serviço me deixou positivamente surpreso. Altamente recomendado!", 'alegria'),
    ("Sinto-me genuinamente feliz por ter escolhido este produto. É exatamente o que eu precisava.", 'alegria'),
    ("Que descoberta maravilhosa! Este produto não só me trouxe alegria, mas também facilitou minha vida de maneira notável.", 'alegria'),
]

frases_preocupacao = [
    ("Estou bastante preocupado com a qualidade deste produto; não atendeu às minhas expectativas.", 'preocupação'),
    ("Que experiência assustadora! Este serviço deixou muito a desejar e gerou insegurança.", 'preocupação'),
    ("Estou realmente apreensivo com a eficácia deste produto. Não parece ser seguro para mim.", 'preocupação'),
    ("A preocupação é grande ao constatar que este serviço não só é ineficiente, mas também complicado de usar.", 'preocupação'),
    ("Este produto é uma decepção! Não cumpre o que promete e isso me causa medo.", 'preocupação'),
    ("A insegurança é palpável ao constatar que o investimento neste serviço foi um erro.", 'preocupação'),
    ("Não consigo conter minha preocupação ao desaconselhar este produto. Decepcionante!", 'preocupação'),
    ("A falta de confiabilidade deste serviço me deixou positivamente assustado. Não recomendaria.", 'preocupação'),
    ("Sinto-me genuinamente preocupado por ter escolhido este produto. Não é o que eu esperava.", 'preocupação'),
    ("Que descoberta desanimadora! Este produto não só me causou medo, mas também complicou minha vida de maneira notável.", 'preocupação')
]

frases_surpresa = [
    ("Estou completamente surpreso com a qualidade deste produto; superou todas as minhas expectativas!", 'surpresa'),
    ("Que surpresa incrível! Este serviço não só atendeu, mas ultrapassou minhas necessidades.", 'surpresa'),
    ("Estou verdadeiramente surpreso com a eficácia deste produto. Fez uma diferença notável para mim.", 'surpresa'),
    ("A surpresa é imensa ao constatar que este serviço não só é eficiente, mas também surpreendentemente fácil de usar.", 'surpresa'),
    ("Este produto é uma surpresa maravilhosa! Cumpre tudo o que promete e mais um pouco.", 'surpresa'),
    ("A surpresa é palpável ao constatar que o investimento neste serviço foi mais do que válido.", 'surpresa'),
    ("Não consigo conter minha surpresa ao recomendar este produto. Simplesmente incrível!", 'surpresa'),
    ("A qualidade excepcional deste serviço me deixou positivamente surpreso. Altamente recomendado!", 'surpresa'),
    ("Sinto-me genuinamente surpreso por ter escolhido este produto. É exatamente o que eu precisava.", 'surpresa'),
    ("Que descoberta surpreendente! Este produto facilitou minha vida de maneira notável.", 'surpresa')
]

frases_raiva = [
    ("Estou extremamente irritado com a péssima qualidade deste produto; não atendeu às minhas expectativas!", 'raiva'),
    ("Que experiência frustrante! Este serviço deixou muito a desejar e gerou muita irritação.", 'raiva'),
    ("Estou verdadeiramente enfurecido com a ineficácia deste produto. Foi uma total perda de tempo para mim.", 'raiva'),
    ("A raiva é imensa ao constatar que este serviço não só é ineficiente, mas também complicado de usar.", 'raiva'),
    ("Este produto é uma fonte de raiva! Não cumpre o que promete e isso me deixa furioso.", 'raiva'),
    ("A irritação é palpável ao constatar que o investimento neste serviço foi um erro colossal.", 'raiva'),
    ("Não consigo conter minha raiva ao desaconselhar este produto. Totalmente decepcionante!", 'raiva'),
    ("A falta de confiabilidade deste serviço me deixou positivamente enfurecido. Não recomendaria de jeito nenhum.", 'raiva'),
    ("Sinto-me genuinamente furioso por ter escolhido este produto. Não é o que eu esperava de forma alguma.", 'raiva'),
    ("Que descoberta revoltante! Este produto não só me causou raiva, mas também complicou minha vida de maneira notável.", 'raiva')
]

base = frases_alegria + frases_preocupacao + frases_raiva + frases_surpresa
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
def fazstemmer(texto):
    """
    Deixa apenas os radicais das palavras
    """
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p))
                       for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming
frasescomstemming = fazstemmer(base)
#print(frasescomstemming)

def buscapalavras(frases):
    """
    Busca as palavras nas frases e separa das emoções
    """
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras
palavras = buscapalavras(frasescomstemming)
#print(palavras)

def buscafrequencia(palavras):
    """
    Define a frequência que as palavras aparecem no banco de dados
    """
    palavras = nltk.FreqDist(palavras)
    return palavras
frequenciatreinamento = buscafrequencia(palavras)
#print(frequenciatreinamento.most_common(50))

def busca_palavrasunicas(frequencia):
    """
    Cria um dicionário de palavras únicas
    """
    freq = frequencia.keys()
    return freq
palavrasunicas = busca_palavrasunicas(frequenciatreinamento)
#print(palavrasunicas)

def extraipalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas
caracteristicasfrase = extraipalavras(['est', 'produto', 'experi'])
#print(caracteristicasfrase)
basecompleta = nltk.classify.apply_features(extraipalavras, frasescomstemming)
print(basecompleta[15])

classificador = nltk.NaiveBayesClassifier.train(basecompleta)
# print(classificador.labels())
# print(classificador.show_most_informative_features(30))
"""
Most Informative Features
                     cap = False           raiva : alegri =      2.6 : 1.0
                   celul = False           raiva : alegri =      1.4 : 1.0
                       a = False          alegri : raiva  =      1.2 : 1.0
                   difer = False          alegri : raiva  =      1.2 : 1.0
                   mater = False          alegri : raiva  =      1.2 : 1.0

Explicando o most Informative Features:

Primeira linha quando a palavra cap= false, ou seja, não aparece significa que tem a probabilidade de 2.6 vezes mais de ser uma frase de raiva
do que de alegria.

Segunda linha quando a palavra celul= false, não aparece significa que tem a probabilidade de 1.4 vezes mais de ser uma frase de raiva
do que de alegria
"""
#Testando com uma frase nova

teste = 'Que descoberta surpreendente e maravilhosa! Este produto facilitou minha vida de maneira notável.'
testestem = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestem.append(str(stemmer.stem(comstem[0])))

nova_frase = extraipalavras(testestem)

distribuicao = classificador.prob_classify(nova_frase)
print('-----------------------')
for classe in distribuicao.samples():
    print("%s: %f" % (classe, distribuicao.prob(classe)))

# Melhorar a base e produzir testes condizentes

