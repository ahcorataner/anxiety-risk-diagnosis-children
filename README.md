
````markdown
# 🧠 SISTEMA DE APOIO AO DIAGNÓSTICO DO RISCO DE TRANSTORNO DE ANSIEDADE EM CRIANÇAS

Este repositório contém o código-fonte desenvolvido para o estudo:  

**“SISTEMA DE APOIO AO DIAGNÓSTICO DO RISCO DE TRANSTORNO DE ANSIEDADE EM CRIANÇAS”**  
(Correspondente ao artigo submetido ao CILAMCE 2025).

---

## 📂 Conteúdo do repositório

O código implementa:

- **Pré-processamento de dados**: leitura, tratamento e separação de features e target.  
- **Divisão treino/teste** estratificada.  
- **Treinamento de modelos**: Random Forest, SVM e MLP.  
- **Ajuste de hiperparâmetros** usando `GridSearchCV`.  
- **Avaliação de desempenho**: acurácia, sensibilidade e especificidade.  
- **Visualizações**: tabelas, gráficos de barras comparativos e matrizes de confusão.

> ⚠️ **Observação:** A base de dados utilizada no estudo **não está incluída neste repositório**.  
> Ela está disponível em:  
> [1] K.L.H. Carpenter, "Preschool Anxiety Lab – Harvard Dataverse", 2016.  
> Available at: [https://doi.org/10.7910/DVN/4X8HAY](https://doi.org/10.7910/DVN/4X8HAY). Accessed on: May 17, 2025.

---

## ⚙️ Instalação e uso

1. Instale as dependências necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
````

2. Ajuste o caminho da base de dados no script:

```python
path = r"C:\caminho\para\a\base.xlsx"
```

3. Execute o script principal:

```bash
python hiperparametros.py
```

4. Os resultados serão gerados como arquivos Excel (`.xlsx`) e imagens (`.png`) no mesmo diretório.

---

## 📌 Aviso de uso

⚠️ **Atenção:**
O código está **disponível apenas para visualização e referência acadêmica**.
**Qualquer uso, redistribuição ou modificação requer autorização prévia da autora.**

Pesquisadores interessados podem solicitar acesso enviando e-mail para:
**[renata.rocha@discente.ufma.br](mailto:renata.rocha@discente.ufma.br)**

---

## 📊 Exemplos de saída

* Tabelas de métricas de desempenho (acurácia, sensibilidade, especificidade).
* Gráficos comparativos de desempenho entre modelos.
* Matrizes de confusão por modelo.

---

## 📝 Referência do estudo

Carpenter et al., 2016. Base de dados disponível em Harvard Dataverse: [https://doi.org/10.7910/DVN/4X8HAY](https://doi.org/10.7910/DVN/4X8HAY)

---

## ✉️ Contato

**Renata Costa Rocha**
[renata.rocha@discente.ufma.br](mailto:renata.rocha@discente.ufma.br)

```

