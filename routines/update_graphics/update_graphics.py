import pandas as pd 
import numpy as np 
from functools import reduce
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import torch
from scipy import spatial
import plotly.graph_objects as go

#Limpeza: preencher apenas o que for atualizar
#lula = pd.read_csv('')
#bolsonaro = pd.read_csv()
#andre = pd.read_csv('')
#simone = pd.read_csv('') 
#felipe = pd.read_csv('')
#ciro = pd.read_csv('')
#bivar = pd.read_csv('')
#leo = pd.read_csv('')
#pablo = pd.read_csv('')
#sofia = pd.read_csv('')
#eymael = pd.read_csv('')
#vera = pd.read_csv('')

#adicionar o nome dos candidatos que for atualizar:
candidatos = []

df = reduce(lambda  left,right: pd.merge(left,right,
                                            how='outer'), candidatos)

df['clean_text'] = df['text'].copy(deep = True)

def remove_special_tags(text): 
  text = unicodedata.normalize("NFKD", text)
  return text  
df['clean_text'] = df.clean_text.apply(remove_special_tags)


df['mentions'] = np.nan
for i in df.index:
    if len(re.findall('@\w+',df['text'][i])) != 0:
        df['mentions'][i] = re.findall('@\w+',df['text'][i])
def remove_mentions(text):
  text = re.sub('@\w+', '', text)
  return text  
df['clean_text'] = df.clean_text.apply(remove_mentions)

def remove_breakline(text):
  text = re.sub('\\n', '', text)
  return text
df['clean_text'] = df.clean_text.apply(remove_breakline)

def remove_links(text):
  text = re.sub('https:.+$', '', text)
  return text
df['clean_text'] = df.clean_text.apply(remove_links)

df['hashtags'] = np.nan
for i in df.index:
    if len(re.findall('#\w+',df['clean_text'][i])) != 0:
        df['hashtags'][i] = re.findall('#\w+',df['clean_text'][i])
def remove_hashtags(text):
  text = re.sub('#\w+', '', text)
  return text
df['clean_text'] = df.clean_text.apply(remove_hashtags)

df['emojis'] = np.nan
emojis = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\U0001F991"
                      "]+", re.UNICODE)
for i in df.index:
    if len(re.findall(emojis,df['text'][i])) != 0:
        df['emojis'][i] = re.findall(emojis,df['text'][i])
#por algum motivo o emoji de lula não quer funcionar no compile, então eu vou ter que fazer à parte
for i in df.index:
    if len(re.findall(u"\U0001F991",df['text'][i])) != 0:
        df['emojis'][i] = re.findall(u"\U0001F991",df['text'][i])
#substituindo o emoji da bandeira do brasil por um aleatório para não atrapalhar posteriormente
bandeira_brasil = re.compile(u"\U0001F1E7\U0001F1F7", re.UNICODE)
for i in df.index:
  if type(df['emojis'][i]) != float:
    df['emojis'][i] = re.sub(bandeira_brasil, u"\U0001F47E", str(df['emojis'][i]))
for i in df.index:
  if type(df['emojis'][i]) == str:
    df['emojis'][i] = eval(df['emojis'][i])
def remove_emojis(text):
    return emojis.sub(r'',text)
df['clean_text'] = df.clean_text.apply(remove_emojis)
squid = re.compile("[" u"\U0001F991""]+", re.UNICODE)
def remove_squid(text):
  return squid.sub(r'',text)
df['clean_text'] = df.clean_text.apply(remove_squid)

df['clean_text'] = df['clean_text'].apply(word_tokenize) 

#Bert
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

def avg_feature_vector(words):
  input_ids = tokenizer.encode(words, return_tensors='pt')
  with torch.no_grad():
      outs = model(input_ids)
      encoded = outs[0][0, 1:-1].numpy()
      mean = np.mean(encoded, axis = 0)
  return mean

def avg_tweets(candidato, lista_keywords):
    feature_vec = np.zeros((768, ), dtype='float32')
    n_tweets = 0
    index_list = []
    for keyword in lista_keywords:
      for i in df.index:
        if i not in index_list:
          if len(re.findall(keyword, ' '.join(df['clean_text'][i])))!= 0:
              if candidato in df['user_name'][i]:
                  n_tweets +=1
                  feature_vec = np.add(feature_vec, avg_feature_vector(' '.join(df['clean_text'][i])))
                  index_list.append(i)
    if (n_tweets > 0):
        feature_vec = np.divide(feature_vec, n_tweets)
    return feature_vec, n_tweets

temas = ['Gasolina', 'Inflação', 'Educação', 'Vacina', 'Dólar', 'Aposentadoria', 'Violência', 'Religião', 'Água', 'CLT', 'Covid', 'LGBT', 'Auxílio Brasil', 'Fome', 'PEC Kamikaze']
embeddings_2012 = pd.read_csv('../presidenciaveis_script/data/embeddings_2012.csv', index_col = 'Unnamed: 0')
tweets_2012 = pd.read_csv('../presidenciaveis_script/data/n_tweets_2012.csv', index_col = 'Unnamed: 0')
embeddings_att = pd.DataFrame(index = embeddings_2012.index, columns = embeddings_2012.columns)
tweets_att = pd.DataFrame(index = tweets_2012.index, columns = tweets_2012.columns)
new_embeddings = pd.DataFrame(index = embeddings_2012.index, columns = embeddings_2012.columns)
new_tweets = pd.DataFrame(index = tweets_2012.index, columns = tweets_2012.columns)

def calcular_embeddings(lista_keywords, coluna):
  for i in new_embeddings.index:
    array = avg_tweets(i, lista_keywords)
    new_embeddings[coluna][i] = array[0]
    new_tweets[coluna][i] = array[1]

calcular_embeddings(['vacina(?i)', 'vacina\S{2}o(?i)', 'imuniza\S{2}o(?i)'],'Vacina')
calcular_embeddings(['aposentadoria(?i)', 'previd\Sncia(?i)'],'Aposentadoria')
calcular_embeddings(['infla\S{2}o(?i)'],'Inflação')
calcular_embeddings(['educa\S{2}o(?i)', 'ensino(?i)'],'Educação')
calcular_embeddings(['d\Slar(?i)'],'Dólar')
calcular_embeddings(['viol\Sncia(?i)', 'ataque(?:s)?(?i)', 'atentado(?i)'],'Violência')
calcular_embeddings(['religi\So(?i)','cren\S{1}(?i)', '\bf\S{1}(?i)'],'Religião')
calcular_embeddings(['(a|á)gua(?i)'],'Água')
calcular_embeddings(['CLT(?i)', 'trabalho(?i)', 'emprego(?i)', 'trabalhador(?:es)?'],'CLT')
calcular_embeddings(['covid(?i)','coronav\S{1}rus(?i)', 'pandemia(?i)', 'quarentena(?i)'],'Covid')
calcular_embeddings(['lgbt(?i)', 'gay(?:s)?(?i)', 'transfobia(?i)', 'homofobia(?i)', 'l\Ssbica(?:s)?(?i)', 'bissexual(?i)', 'transsexual(?i)'],'LGBT')
calcular_embeddings(['fome(?i)'],'Fome')
calcular_embeddings(['gasolina(?i)'],'Gasolina')
calcular_embeddings(['aux\Slio\sbrasil(?i)','bolsa\sfam\Slia(?i)'],'Auxílio Brasil')
calcular_embeddings(['PEC\sKamikaze(?i)'],'PEC Kamikaze')

#Atualizar os embeddings já calculados
def add_comma(match):
    return match.group(0) + ','

for column in embeddings_2012.columns:
  for index in embeddings_2012.index:
    embeddings_2012[column][index] =  re.sub(r'(\d(.)?)\s',add_comma, embeddings_2012[column][index])
  embeddings_2012[column] = embeddings_2012[column].apply(eval)

for column in new_embeddings.columns:
  for index in new_embeddings.index:
    if np.isnan(new_embeddings[column][index]).all():
      new_embeddings[column][index] = list(np.zeros((768, ), dtype='float32'))

new_tweets.fillna(value = 0, inplace = True)

for i in embeddings_2012.index:
  for j in embeddings_2012.columns:
    embeddings_att[j][i] = np.divide(np.add(np.array([(float(x) * tweets_2012[j][i]) for x in embeddings_2012[j][i]]),
                                            np.array([(float(x) * new_tweets[j][i]) for x in new_embeddings[j][i]])),
                                            tweets_2012[j][i]+ new_tweets[j][i],
                                            where = tweets_2012[j][i]+ new_tweets[j][i]!=0)
for i in tweets_att.index:
  for j in tweets_att.columns:
    tweets_att[j][i] = tweets_2012[j][i] + new_tweets[j][i]

embeddings_att.to_csv("../presidenciaveis_script/data/embeddings_att.csv")
tweets_att.to_csv("../presidenciaveis_script/data/tweets_att.csv")

def similaridade_candidatos(candidato1, candidato2, tema):
    cand1 = np.asarray(embeddings_att[tema][candidato1], dtype='float64')
    cand2 = np.asarray(embeddings_att[tema][candidato2], dtype='float64')
    sim = 0
    if np.all((cand1 == 0))==False and np.all((cand2 == 0))==False:
      sim =  1 - spatial.distance.cosine(cand1, cand2)
    return sim

def correlacao(tema):
    similarity_array = np.zeros((len(embeddings_att.index),len(embeddings_att.index)))
    lista_legendas_celulas = []

    for i in range(len(embeddings_att.index)):
      for j in range(len(embeddings_att.index)):
        similarity_array[i][j] = round(similaridade_candidatos(embeddings_att.index[i], embeddings_att.index[j],tema), 2)
      
    similarity_array = np.ma.masked_equal(np.tril([x for x in similarity_array],-1),0)

    df2 = pd.DataFrame(similarity_array, index = embeddings_att.index, columns = embeddings_att.index)
    for i in df2.columns:
      if len(df2[i].unique()) == 1 or list(df2[i].dropna().unique()) == [1]:
        df2.drop(i, axis = 1, inplace = True)
    for j in df2.index:
      if list(set(df2.loc[j].dropna())) == [1]:
        df2.drop(j, inplace = True)
    df2.dropna(axis=0, how='all', inplace = True)
    df2.dropna(axis=1, how='all', inplace = True)
    if len(df2.columns) != 1 and len(df2.index) != 1:
      df2 = (df2-df2.min().min())/(df2.max().max()-df2.min().min())
      df2 = df2.round(2).fillna("")

    for i in range(len(df2.index)):
      linha_legenda = []
      for j in range(len(df2.columns)):
        linha_legenda.append(f"Para {df2.index[i]}, há {tweets_att[tema][df2.index[i]]} tweet(s) sobre o assunto. Já para {df2.columns[j]}, há {tweets_att[tema][df2.columns[j]]}.")
      lista_legendas_celulas.append(linha_legenda)

    fig = go.Figure(data=go.Heatmap(colorbar={"title": "Similaridade dos discursos", "titleside": "right", "bordercolor": "#ededf5", "borderwidth":0.5},
                                    x = df2.columns,
                                    y = df2.index,
                                    z = df2,
                                    text=df2,
                                    xgap = 5,
                                    ygap = 5,
                                    hovertext = lista_legendas_celulas,
                                    hoverinfo = 'text',
                                    hoverongaps = False,
                                    texttemplate="%{text}",
                                    textfont={"size":18}, 
                                    colorscale=[[0.0, '#f1863d'], 
                                    [0.5, '#ededf5'], 
                                    [1.0, '#0660a2']],
                                    connectgaps = False,), 
                    layout=go.Layout(title=tema,
                                      title_x=0.5,
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_showgrid=False, yaxis_showgrid=False,
                                      yaxis={"autorange":"reversed"}))

    fig.write_html(f'../presidenciaveis_script/graphic/{tema}.html')

for column in embeddings_att.columns:
  correlacao(column)