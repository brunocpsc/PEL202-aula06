# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:42:47 2020
"""

from sklearn.datasets import load_iris
import numpy as np
import math
inf=math.inf

#carregament do dataset IRIS
iris=load_iris()
X = load_iris().data
y = load_iris().target
#tranforma o dataset em um np array
data=np.column_stack((X, y))

#função que verificar se há apenas um item restante na árvore
def ver_uni_class(data):
    
    uq=np.unique(data[:,4])
    if len(uq)==1:
        return True
    else:
        return False
    
#função que verificar a classe mais predominante
def ver_class(data):
    
    c,ct = np.unique(data[:,4], return_counts=True)
    ct=ct.argmax()
    return c[ct]

#calculo de entropia em duas etapas - quanto menor entropia, maior o ganho
#se calcula a entropia por critério de divisão e depois se executa a ponderação
#a ponderação é realizada para o cálculo da entropia total
def entropia_i(data):
    
    _,prob=np.unique(data[:,4],return_counts=True)
    prob=prob/prob.sum()
    e=sum(prob*-np.log2(prob))

    return e

def entropia_t(data_ac,data_ab):
    
    pab=len(data_ab)/(len(data_ac)+len(data_ab))
    pac=len(data_ac)/(len(data_ac)+len(data_ab))
    et=(pab*entropia_i(data_ab)+pac*entropia_i(data_ac))
    
    return et

#cálculo do critério de divisão nos datasets
#consideramos a linha média entre os pontos
def cri_div(data):
    
    div={}
    _, ncol=data.shape
    
    for i in range(ncol-1):
        div[i]=[]
        v=data[:,i]
        uq=np.unique(v)
        
        for j in range(len(uq)):
            if j!= 0:
                v0=uq[j]
                v1=uq[j-1]
                v2=(v0+v1)/2
                div[i].append(v2)   
    return div


#função que utiliza o critério de divisão e determina a variável de decisão (largura/comprimento das sépatas/pétalas)
#a função retorna, além da variável o valor limítrofe
def escolhe_sep(data,div):
    et=inf
    for i in div:
        for v in div[i]:
            data_ac,data_ab=separa(data,i,v)
            ett=entropia_t(data_ac,data_ab)
            
            if ett <= et:
                et=ett
                mcol=i
                mcri=v           
                
    return mcol,mcri

#função que separa o dataset de acordo com os critérios
def separa(data,col,cri):
    sep=data[:,col]
    data_ac=data[sep>cri]
    data_ab=data[sep<=cri]

    return data_ac,data_ab
    
def ID3(data,i):
    
    #verificar se há apenas uma classe ou se atingiu o nível máximo de iterações
    if (ver_uni_class(data)) or (i==3):
        cl = ver_class(data)
        i+=1
        return cl
    
    else:
        i+=1
        div=cri_div(data)
        mcol,mcri=escolhe_sep(data,div)
        data_ac,data_ab=separa(data,mcol,mcri)
        
        rotulo = "{} <= {}".format(mcol, mcri)
        no = {rotulo: []}
        
        left=ID3(data_ac,i)
        right=ID3(data_ab,i)
        
        if left == right:
            no = left
   
        else:
            no[rotulo].append(left)
            no[rotulo].append(right)
            
    return no

def define_classe(value):
    if value[3]<= 0.8:
        print('Classe 0 - Setosa')
    else:
        if value[3]>1.75:
            print('Classe 2 - Virginica')
        else:
            if value[2]>4.95:
                print('Classe 2 - Virginica')
            else:
                print('Classe 1 - Versicolor')
    return 1


#coleta dos critérios da árvore:
#print(arv['3 <= 0.8'][0])
#print(arv['3 <= 0.8'][1])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'][0])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'][1])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'][1]['2 <= 4.95'])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'][1]['2 <= 4.95'][0])
#print(arv['3 <= 0.8'][0]['3 <= 1.75'][1]['2 <= 4.95'][1])

#chama árvore
arv=ID3(data,0)
#imprime árvore e critérios
print(arv)
#entrada das variáveis dos características da flor
flor_val = list(float(num) for num in input("Entre as variáveis x0,x1,x2 e x3 separadas por espaço: ").strip().split())[:4]
print("Valores: ", flor_val)

#classificaçao da flor
define_classe(flor_val)
