# 🧬 Seleção de atributos com Algoritmo Genético
## Introdução
<p> Este repositório apresenta uma forma de se fazer seleção de atributos (feature selection) para machine learning, usando algoritmo genético. Vou utilizar a  <a href="https://github.com/joao-savietto/genetic-algorithm"> minha própria implementação</a> para exemplificar o seu uso. </p>

### 💭 Por que fazer seleção de atributos?
<p> Quando montamos um dataset para machine learning, coletamos uma quantidade enorme de informações na esperança de haver algo útil para o modelo. A qualidade dos modelos depende totalmente da qualidade dos dados, incluindo a escolha das variáveis. O resultado dos modelos pode ser prejudicado por variáveis que são pouco descritivas ou que produzem ruído. Técnicas de seleção de atributos permitem melhorar o resultado dos modelos e ainda reduzir o tempo de treinamento, inferência e consumo de memória devido à redução da quantidade de dados. </p>

### 🎲 Dataset

<p> O dataset usado é o Student Performance, disponível no Kaggle <a href="https://www.kaggle.com/datasets/whenamancodes/student-performance"> neste link.</a> O objetivo é prever as notas de matemática e português dos alunos com base em algumas variáveis. Os dados contém as notas "G1", "G2" e "G3". Eu foquei esse experimento em prever as notas iniciais (G1). </p>

### 📝 Representação do problema
<p> O algoritmo usado é o LinearRegression do sklearn. Inicialmente, eu treinei o modelo com todos os atributos, exceto G2 e G3, totalizando 30 atributos. O mean squared error (MSE) na validação cruzada (10 folds) foi de 0.037.</p>
<p>A partir disso, perguntamos: <b> como é possível minimizar o MSE utilizando o menor número possível de atributos?</b></p>
<p>A classe <i> Individual </i> foi desenvolvida de forma que testa diferentes combinações de atributos. O fitness é a soma do número de atributos que foram removidos, subtraído o MSE do modelo (em validação cruzada 10-folds). O fitness é definido como zero caso todos os atributos sejam removidos, ou caso o MSE seja superior a 0.0336. </p>

<p><b>Parte do código:</b></p>

```
def calculate_fitness(self):
    self.selected_categorical_features = [self.all_features[i] for i in range(len(self.chromossome)) if self.chromossome[i] == 1 and self.all_features[i] in self.categorical_features]
    self.selected_numeric_features = [self.all_features[i] for i in range(len(self.chromossome)) if self.chromossome[i] == 1 and self.all_features[i] in self.numeric_features]        
    if self.selected_categorical_features + self.selected_numeric_features == []:
        self.fitness = -1
        return -1
    
    self.dataset_pred = pd.DataFrame()
    self.dummy_categorical = pd.DataFrame()
    self.dataset_numeric = pd.DataFrame()

    if self.selected_categorical_features:
        self.dataset_categorical = dataset[self.selected_categorical_features]
        self.encoder = OneHotEncoder(categories='auto')
        self.dummy_categorical = self.encoder.fit_transform(self.dataset_categorical).toarray()
        self.dummy_categorical = pd.DataFrame(self.dummy_categorical, index=self.dataset_categorical.index)  
        self.dataset_pred = self.dummy_categorical
    
    if self.selected_numeric_features:
        self.dataset_numeric = dataset[self.selected_numeric_features]
        
    self.dataset_pred = pd.concat([self.dataset_numeric, self.dummy_categorical], axis=1)        
    self.scaler = MinMaxScaler()
    self.X = self.scaler.fit_transform(self.dataset_pred)
    self.y = np.array(dataset[target])
    self.scaler_y = MinMaxScaler()
    self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1))
    self.model = LinearRegression()      
    self.scores = cross_validate(self.model, self.X, self.y, cv=10, scoring=('neg_mean_squared_error'))['test_score'].mean()
    self.fitness = sum([1 for x in self.chromossome if x == 0]) + self.scores
    if self.scores <= -0.0336:
        self.fitness = self.scores
    return self.fitness
```

<p><i>Versão completa no repositório</i></p>

## 🎯 Resultado
<p> Após 29 iterações, o algoritmo genético encontrou 8 atributos (de um total de 30), e o MSE caiu de 0.037 para 0.033. </p>

<br/>

```
Base MSE:  -0.037870924891187596
Creating initial population...
Done
Starting algorithm
Iteration [ 1  |  100 ] Best fitness: -0.034871438466585594 | Best individual in current generation: -0.034871438466585594
Iteration [ 2  |  100 ] Best fitness: -0.034759942079201724 | Best individual in current generation: -0.034759942079201724
Iteration [ 3  |  100 ] Best fitness: -0.034759942079201724 | Best individual in current generation: -0.03492139437259772
Iteration [ 4  |  100 ] Best fitness: -0.034537089253083256 | Best individual in current generation: -0.034537089253083256
Iteration [ 5  |  100 ] Best fitness: -0.03436318255483341 | Best individual in current generation: -0.03436318255483341
Iteration [ 6  |  100 ] Best fitness: -0.03388410983941494 | Best individual in current generation: -0.03388410983941494
Iteration [ 7  |  100 ] Best fitness: 18.966643539086366 | Best individual in current generation: 18.966643539086366
Iteration [ 8  |  100 ] Best fitness: 18.966643539086366 | Best individual in current generation: -0.03366244273307996
Iteration [ 9  |  100 ] Best fitness: 18.966643539086366 | Best individual in current generation: -0.03364173504022452
Iteration [ 10  |  100 ] Best fitness: 18.966643539086366 | Best individual in current generation: 17.96647411438135
Iteration [ 11  |  100 ] Best fitness: 19.966675188694246 | Best individual in current generation: 19.966675188694246
Iteration [ 12  |  100 ] Best fitness: 19.966675188694246 | Best individual in current generation: 19.966675188694246
Iteration [ 13  |  100 ] Best fitness: 20.966582978615204 | Best individual in current generation: 20.966582978615204
Iteration [ 14  |  100 ] Best fitness: 20.966582978615204 | Best individual in current generation: 20.966582978615204
Iteration [ 15  |  100 ] Best fitness: 20.966582978615204 | Best individual in current generation: 20.966582978615204
Iteration [ 16  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 17  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 18  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 19  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 20  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 20.96647492469942
Iteration [ 21  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 22  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 23  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 24  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 20.96647492469942
Iteration [ 25  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 26  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 27  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 28  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
Iteration [ 29  |  100 ] Best fitness: 21.966694377360938 | Best individual in current generation: 21.966694377360938
No parents found. Stopping algorithm
8 features kept out of 30
Features kept: ['sex', 'Mjob', 'Fjob', 'schoolsup', 'studytime', 'failures', 'goout', 'health']
Base MSE:  -0.037870924891187596
Final MSE: -0.033305622639061276
```