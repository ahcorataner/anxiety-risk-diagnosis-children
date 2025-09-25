\# 🧠 SISTEMA DE APOIO AO DIAGNÓSTICO DO RISCO DE TRANSTORNO DE ANSIEDADE EM CRIANÇAS



Este repositório contém o código-fonte desenvolvido para o estudo:  



\*\*“SISTEMA DE APOIO AO DIAGNÓSTICO DO RISCO DE TRANSTORNO DE ANSIEDADE EM CRIANÇAS”\*\*  

(Correspondente ao artigo submetido ao CILAMCE 2025).



---



\## 📂 Conteúdo do repositório



O código implementa:



\- \*\*Pré-processamento de dados\*\*: leitura, tratamento e separação de features e target.  

\- \*\*Divisão treino/teste\*\* estratificada.  

\- \*\*Treinamento de modelos\*\*: Random Forest, SVM e MLP.  

\- \*\*Ajuste de hiperparâmetros\*\* usando `GridSearchCV`.  

\- \*\*Avaliação de desempenho\*\*: acurácia, sensibilidade e especificidade.  

\- \*\*Visualizações\*\*: tabelas, gráficos de barras comparativos e matrizes de confusão.



> ⚠️ \*\*Observação:\*\* A base de dados utilizada no estudo \*\*não está incluída neste repositório\*\*.  

> Ela está disponível em:  

> \[1] K.L.H. Carpenter, "Preschool Anxiety Lab – Harvard Dataverse", 2016.  

> Available at: \[https://doi.org/10.7910/DVN/4X8HAY](https://doi.org/10.7910/DVN/4X8HAY). Accessed on: May 17, 2025.



---



\## ⚙️ Instalação e uso



1\. Instale as dependências necessárias:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn openpyxl



