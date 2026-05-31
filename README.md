# Breast Cancer Diagnosis — ML Model Comparison

A compact GitHub README for six notebooks that explore classification and clustering on breast-cancer data.

## Dataset
- File: `data (1).csv`
- Target: `diagnosis`
- Mapping: `M -> 1`, `B -> 0`
- Dropped columns: `id`, `Unnamed: 32`
- Split: stratified 30% train / 35% test1 / 35% test2
- Positive class: malignant (`1`)

## Notebooks
- `Sigmoid.ipynb` — logistic regression with sigmoid, gradient descent, L1/L2 regularization
- `Naive.ipynb` — Gaussian Naive Bayes classification
- `SVM.ipynb` — linear, polynomial, and RBF SVM plus GridSearchCV and a recall-oriented SVM
- `KNN.ipynb` — KNN with stratified cross-validation and recall-based model selection
- `Decion_Tress.ipynb` — Decision Tree, Bagging, Random Forest, and XGBoost
- `K_Means6.ipynb` — KMeans, Agglomerative, Divisive, and Gaussian Mixture clustering

## Export figures directly from notebook code
Use the same helper in each notebook:
```python
def save_fig(name):
    plt.savefig(f"assets/{name}", dpi=300, bbox_inches="tight")
```
Call it right after each plot:
```python
save_fig("decision_tree_test1_cm.png")
```
Then embed it in Markdown:
```md
![Decision Tree Test1 CM](assets/decision_tree_test1_cm.png)
```

Suggested figure files:
<img width="1356" height="767" alt="image" src="https://github.com/user-attachments/assets/82389b80-e9b5-4a0e-862c-b7c4f7db9cec" />
<img width="1359" height="766" alt="image" src="https://github.com/user-attachments/assets/0fb1c0b9-36c6-4b97-956b-a8acbd873440" />

<img width="1359" height="767" alt="image" src="https://github.com/user-attachments/assets/93303ad7-50a1-4a13-a663-554ff1a4068b" />
<img width="1359" height="767" alt="image" src="https://github.com/user-attachments/assets/836ffb79-ea8a-4289-b007-1d8eec27a7b7" />
<img width="1311" height="417" alt="image" src="https://github.com/user-attachments/assets/b6ced39c-8d8a-4557-9ccb-4ce54abcc22f" />
<img width="1357" height="766" alt="image" src="https://github.com/user-attachments/assets/101960b5-4902-4b54-8a83-75d6c74d0f70" />
<img width="1359" height="747" alt="image" src="https://github.com/user-attachments/assets/a6435ed9-3e4e-490c-b553-eac5057f220f" />
<img width="1351" height="767" alt="image" src="https://github.com/user-attachments/assets/f0fec3e0-f753-444f-b3ca-abf181724453" />
<img width="1306" height="719" alt="image" src="https://github.com/user-attachments/assets/089f87fe-565c-4bf4-8a86-02ea95b6abc7" />
<img width="1352" height="767" alt="image" src="https://github.com/user-attachments/assets/7d32da12-0547-43b0-be2e-9fbbef688c67" />
<img width="1357" height="765" alt="image" src="https://github.com/user-attachments/assets/0e6fa8ba-8dd3-4776-9c6c-e290761497da" />
<img width="1357" height="765" alt="image" src="https://github.com/user-attachments/assets/8854d110-175e-4a75-a695-513a04ee77d0" />
![Uploading image.png…]()

## Performance summary

### `Sigmoid.ipynb`
- Train: Acc 0.9000 | Prec 0.8382 | Rec 0.9048 | F1 0.8702
- Test1: Acc 0.9196 | Prec 0.8919 | Rec 0.8919 | F1 0.8919
- Test2: Acc 0.9000 | Prec 0.8481 | Rec 0.8933 | F1 0.8701
- Notes: logistic regression with selective standardization and L1/L2 tuning.

### `Naive.ipynb`
- Train: Acc 0.9294 | Prec 0.9180 | Rec 0.8889 | F1 0.9032
- Test1: Acc 0.9497 | Prec 0.9444 | Rec 0.9189 | F1 0.9315
- Test2: Acc 0.9350 | Prec 0.9306 | Rec 0.8933 | F1 0.9116
- Notes: a strong lightweight baseline.

### `Decion_Tress.ipynb`

#### Decision Tree
- Train: Acc 0.9765 | Prec 0.9836 | Rec 0.9524 | F1 0.9677
- Test1: Acc 0.9146 | Prec 0.8904 | Rec 0.8784 | F1 0.8844
- Test2: Acc 0.8900 | Prec 0.8354 | Rec 0.8800 | F1 0.8571
- Notes: tuned with GridSearchCV; very strong fit on train, weaker generalization.

#### Bagging
- Train: Acc 0.9706 | Prec 0.9833 | Rec 0.9365 | F1 0.9593
- Test1: Acc 0.9698 | Prec 0.9722 | Rec 0.9459 | F1 0.9589
- Test2: Acc 0.9350 | Prec 0.9429 | Rec 0.8800 | F1 0.9103
- Notes: improved stability over the single tree.

#### Random Forest
- Train: Acc 0.9706 | Prec 0.9833 | Rec 0.9365 | F1 0.9593
- Test1: Acc 0.9598 | Prec 0.9853 | Rec 0.9054 | F1 0.9437
- Test2: Acc 0.9300 | Prec 0.9420 | Rec 0.8667 | F1 0.9028
- Notes: strong generalization across all splits.

#### XGBoost
- Train: Acc 1.0000 | Prec 1.0000 | Rec 1.0000 | F1 1.0000
- Test1: Acc 0.9900 | Prec 1.0000 | Rec 0.9730 | F1 0.9863
- Test2: Acc 0.9500 | Prec 0.9577 | Rec 0.9067 | F1 0.9315
- Notes: best overall classifier in the saved runs.

### `SVM.ipynb`

#### Linear / tuned SVM
- Train: Acc 0.9770 | Prec 1.0000 | Rec 0.9403 | F1 0.9692
- Test1: Acc 0.9749 | Prec 0.9859 | Rec 0.9459 | F1 0.9655
- Test2: Acc 0.9500 | Prec 0.9710 | Rec 0.8933 | F1 0.9306
- Notes: strongest non-ensemble run; GridSearchCV selected the best configuration.

#### Polynomial kernel
- Test1: Acc 0.9146 | Prec 1.0000 | Rec 0.7703 | F1 0.8702
- Notes: very high precision, but lower recall.

#### RBF kernel
- Test1: Acc 0.9045 | Prec 0.9825 | Rec 0.7568 | F1 0.8550
- Notes: precision stays high, recall drops.

#### Recall-oriented SVM
- Train: Acc 0.9706 | Prec 0.9531 | Rec 0.9683 | F1 0.9606
- Test1: Acc 0.9447 | Prec 0.9315 | Rec 0.9189 | F1 0.9252
- Test2: Acc 0.9300 | Prec 0.9552 | Rec 0.8533 | F1 0.9014
- Notes: tuned to catch malignant cases better.

### `KNN.ipynb`
- Train: Acc 1.0000 | Prec 1.0000 | Rec 1.0000 | F1 1.0000
- Test1: Acc 0.9548 | Prec 0.9851 | Rec 0.8919 | F1 0.9362
- Test2: Acc 0.9400 | Prec 0.9846 | Rec 0.8533 | F1 0.9143
- Notes: best K chosen by recall via StratifiedKFold; perfect train score suggests some overfitting.

### `K_Means6.ipynb`
- KMeans silhouette favored `K = 2` (~0.334), with `K = 3` very close (~0.333), then a clearer drop at `K = 4` and `K = 5`.
- Other methods: Agglomerative clustering, divisive clustering via bisecting KMeans, and GMM.
- Visuals: elbow, silhouette, dendrograms, and t-SNE projections.
- Notes: this notebook is unsupervised, so quality is judged by cluster separation rather than accuracy.

## Key takeaways
- XGBoost was the strongest overall classifier in the saved runs.
- Linear/tuned SVM, Bagging, Random Forest, and KNN were also strong.
- Naive Bayes was a solid lightweight baseline.
- Decision Tree overfit more than the ensemble models.
- KNN reached perfect training accuracy, so keep an eye on overfitting.
- Clustering is exploratory and best interpreted with silhouette scores and t-SNE.

## How to run
1. Put `data (1).csv` in the same folder as the notebooks, or update the file paths.
2. Install: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `xgboost`, `jupyter`.
3. Open any notebook in Jupyter or VS Code.
4. Run cells from top to bottom.
5. Save figures into `assets/` before pushing to GitHub.

## Repository structure
- `README.md`
- `Sigmoid.ipynb`
- `Decion_Tress.ipynb`
- `K_Means6.ipynb`
- `Naive.ipynb`
- `SVM.ipynb`
- `KNN.ipynb`
- `data (1).csv`
- `assets/` for exported plots

## GitHub image examples
- `![Sigmoid Confusion Matrix](assets/sigmoid_confusion_matrix.png)`
- `![Naive Confusion Matrix](assets/naive_confusion_matrix.png)`
- `![Decision Tree Comparison](assets/decision_tree_test1_cm.png)`
- `![SVM Grid Search](assets/svm_grid_search_results.png)`
- `![KNN CV Plot](assets/knn_cv_metrics.png)`
- `![KMeans Silhouette Plot](assets/kmeans_silhouette.png)`

## Final note
This version is short enough for GitHub, but still keeps the main performance results and image hooks. Export the figures once, commit them under `assets/`, and the Markdown links will render automatically.
