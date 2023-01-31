# Token-level Word Sense Disambiguation for transformer models

_Unfinished work in progress..._


Transformer models trained on massive amounts of unlabeled raw data demonstrate state-of-the-art performance in most if not all NLP tasks, surpassing human-level ability in many of them. At the same time such models still show potential for improvement on tasks where relational information among concepts plays important role, for example HyperLex. This gap can be filled once an effective way to induce information contained in existing relational sources, carefully crafted by humans, can be found.

With a goal to research and implement a method to enrich transformer models, like **Bert**, with the relational knowledge contained in knowledge graphs like, Princeton **WordNet**, and ontologies like, **DBPedia** and **OpenCyc** for general knowledge as well as **FIBO**, specific to financial services domain, in such a way that original transformer model retains it universal capabilities for token- and sequence-level classification with improved accuracy and facilitates further pretraining on downstream tasks, an attempt was made to implement a mixture of 'word-expert' and 'full-vocabulary' approaches for the task of **word sense disambiguation** in transformer models at token level allowing the full disambiguated sense sequence to be further processed.

The key idea is to limit classification task for each token to a set of WordNet senses this token can actually represent versus classifying against full sense inventory. Such approach allows to decrease classifications space three orders of magnitude from more than 175000 synsets in WordNet to around 200 for each token, with 200 being chosen as a hyperparameter after initial empirical data exploration. These class identifiers represent particular sense for each token and are not aligned among tokens, effectively turning single classification task into `[number of tokens in tokenizer vocabulary] * 200` smaller classification tasks. It also allows to account for synsets consisting of multiple tokens, especially for verbs like 'run out (of time, resources...)' by tagging all participating tokens with shared synset. This is achieved by retrieving all lemmas for each synset, enumerating their possible surface forms and tokenizing by model-specific tokenizer keeping mapping between all produced tokens and that synset. Finally, tokens are filtered out based on idea that if a single token participates in too many synsets (particular value for each part of speech is specified as another hyperparameter) it plays purely syntactic role and can be safely eliminated from classification task.

Once such token sense vocabulary is built, it is used to extract contextual vector representation from output layers of the specific transformer model for each tagged surface form in training corpora (SemCor and WNGC in this experiment) to build training and validation datasets in order to save computational resources by caching these contextual vectors. Same procedure is applied to test corpora.

### Network architecture overview

The most simple architecture to implement solution for this problem was chosen to include a simple fully connected classification network shared among all tokens which receives contextual hidden representation for a token from the underlying transformer model and an embedding layer with specific learned value for each token to compensate for mis-alignment among token class identifiers. During final testing a heuristic approach is used to combine predicted senses from multiple tokens representing a full word from test corpora is used, which will be unnecessary once model architecture is changed to the transformer architecture in further experiments.

Three ways of combining token contextual representation with dimension `ctx_dim` and learnable token embedding with dimension `emb_dim` were estimated:

* appending embedding vector to contextual vector and using this extended representations as input to classification network (input dimension being `ctx_dim + emb_dim`)
* multiplying contextual vector by embedding vector (input dimension being `ctx_dim`)
* implementing CNN-filter-like solution treating embedding as several token-specific filters for `ctx_dim`-channels contextual vector of length 1. Variant with 3 filters was tried resulting in input dimension of 3 values to classification network, but due to significantly increased computation burden and relatively low performance compared to former options, this options was not researched deeper.

Due to such architecture and preference for large batch size (for performance reasons) the decision was made to introduce **separate learning rates** for token-specific embedding layer and fully connected classification layers shared by all tokens because each token's embedding is encountered only a few times in a large batch, while classification layers accumulate gradient for each example. This decision proved to be useful in hyperparameter search and optimization experiments where the best results were obtained when these learning rates differed one order of magnitude, and also in PBT where it allows more fine-tuning. Similar difference is observed in plots of grad norm for embedding layer (0-th) and first fully connected layer:

![grad_norm_per_layer.png](img/grad_norm_per_layer.png?fileId=134438#mimetype=image%2Fpng&hasPreview=true)
Interesting behavior was observed in relation between layers and performance metrics, where correlation between embedding layer learning rate and recall performance was higher that with precision performance, while classification layer's learning rate demonstrated opposite behavior. This phenomenon was exploited in experiment with separate learning rate schedulers for each parameter group monitoring different metrics, which resulted in faster network training. 

### Training data overview

Class histogram of training data including 0-th / default sense:

![class_histogram_with_0.png](img/class_histogram_with_0.png?fileId=134332#mimetype=image%2Fpng&hasPreview=true)

Class histogram of training data excluding 0-th / default sense:

![class_histogram_without_0.png.png](img/class_histogram_without_0.png.png?fileId=134331#mimetype=image%2Fpng&hasPreview=true)

Prepared training dataset exhibits clear **disproportion between number of examples** for each class. Such disproportion is **natural** due to some senses being much more widespread in training corpora while others are less so. Due to decision to tag every token with at least one sense meaning 'no particular sense' for tokens like punctuation etc. to increase precision of the model, this sense's class (codified as 0-th) strongly dominates all other classes in the dataset. Initial experiments showed tendency of the model to initially learn to predict this class for all tokens easily obtaining notable precision and recall scores. Adding **custom weighting for classes in Cross-Entropy loss** function helped to mitigate this issue with option of decreasing weight of only 0-th class showing best results due keeping intact natural distribution of senses while complete balancing of the dataset moved most frequently predicted sense towards arithmetic mean of number of senses.

Histograms of classes predicted by the model during first 4 training epochs for equal class weights set to 1; all class equal to 1 except 0-th set to 0.1 weight; balanced (each class weight computed proportionally to its presence in training data):

![predictions_histogram_equal_CE.png](img/predictions_histogram_equal_CE.png?fileId=134393#mimetype=image%2Fpng&hasPreview=true)![predictions_histogram_depress_zero_CE.png](img/predictions_histogram_depress_zero_CE.png?fileId=134386#mimetype=image%2Fpng&hasPreview=true)![predictions_histogram_balanced_CE.png](img/predictions_histogram_balanced_CE.png?fileId=134378#mimetype=image%2Fpng&hasPreview=true)

### Confusion matrix

Confusion matrix shows that model clearly distinguishes between parts of speech and most misclassification error happens inside one part of speech with more errors towards senses less presented in training data, with verbs being relatively harder than other parts of speech. Often mentioned too fine granularity of senses in WordNet as discussed in other works, may also contribute to observed behavior.

![confusion_matrix.png](img/confusion_matrix.png?fileId=134352#mimetype=image%2Fpng&hasPreview=true)

### Hyperparameter search / optimization

**Ray Tune** allows to use several methods for hyperparameter search and optimization ranging from classical **grid search** across search space, which is applicable when space is not very large and enough computation resources are available. When search space is significantly larger or computational resources are expensive, **sampling parameter combinations** from full search space using parameter **search algorithm** taking into account previous sample results, allows to find well performing parameter combinations much faster. **Early stopping** of suboptimal trials using **ASHA** algorithm allows to save computational resources and decrease overall experiment time significantly. Such techniques work best for architectural hyperparameters, while optimization-time parameters (i.e. which can be changed for a model not interrupting training process, like learning rate adjustment) can leverage **Population Based Training** technique, which goes beyond simple learning rate scheduling, allowing to change several optimization-time parameters during training by sampling from search space or increasing/decreasing by a constant factor, always choosing best performing trial to continue parameter mutations. Such approach can help to overcome some common problems with sticking to local minimum/maximum more effectively.

**F-measure** with equal preference for precision and recall was chosen as optimization objective due to its more accurate representation of classification performance in a multi-class classification problem compared with simple classification accuracy.

Full **grid search** with restricted set of hyperparameter choices didn't s show any unexpected results, proving that wider layers and larger embeddings translate into better performance, however 1 layer deep network showed better performance than a 5 layer deep network, which is explained by only two epochs chosen for search experiment duration due to computation budget which is clearly insufficient for a deeper network training. Top 9 performing trials shown below:

![grid_scatter_top.png](img/grid_scatter_top.png?fileId=134487#mimetype=image%2Fpng&hasPreview=true)![grid_parallel_top.png](img/grid_parallel_top.png?fileId=134488#mimetype=image%2Fpng&hasPreview=true)Computationally comparable number of epochs consumed by **sampling parameters** using **OnePlusOne search algorithm with ASHA** scheduler not only allows to explore significantly larger parameter space (two orders of magnitude larger in this experiment) illuminating some inter-parameter patterns, like learning rate difference for different layers, but also achieves around 60% higher overall F1 score giving more realistic picture of potential model performance:

![sample_scatter_all.png](img/sample_scatter_all.png?fileId=134545#mimetype=image%2Fpng&hasPreview=true)![sample_parallel_all.png](img/sample_parallel_all.png?fileId=134546#mimetype=image%2Fpng&hasPreview=true)![sample_parallel_top.png](img/sample_parallel_top.png?fileId=134537#mimetype=image%2Fpng&hasPreview=true)

From engineering point of view Ray Tune provides very convenient **resource management** for trial execution even running on a single machine, not in a cluster, allowing to share same GPU among several trials for increased throughput, especially for experiments where data-loading processes consume significant time preparing data, what can be clearly observed with a **profiler**.

### Forbidden Testing:

To estimate overall viability of chosen approach some preliminary tests were run against several widely used test corpora for WSD task following Raganato et al. framework. Obtained results showed contradictory trends - accuracy for verbs, which is usually the lowest, is around 50% showing the least gap among all parts of speech with other published results, while nouns, which usually achieve accuracy of around 75%, scored only around 40%. Such low result for nouns needs further investigation and is most certainly related to hyperparameters used during building token-sense vocabulary. Meanwhile, results obtained on test set were very close to those obtained on validation set during regular experiments allowing to focus tuning on validation set.

| POS\Corpus | Senseval-2 | Senseval-3 | SemEval-07 | SemEval-13 | SemEval-15 | All  |
|------------|-----------:|-----------:|-----------:|------------|------------|------|
| noun       | 47.8       | 40.8       | 43.3       | 36.7       | 44.1       | 40.9 |
| verb       | 44.6       | 49.1       | 52.8       |     -      | 43.6       | 47.6 |
| adjective  | 62.2       | 51.4       |     -      |     -      | 51.5       | 55.8 |
| adverb     | 58.5       | 25.7       |     -      |     -      | 66.3       | 58.4 |
| all        | 51.0       | 45.1       | 49.6       | 36.7       | 46.8       | 50.0 |

Taking into account present simplicity of the architecture and limited training of the best model, the work is expected to achieve higher results in the future.

### References:

* Raganato, A., Camacho-Collados, J. and Navigli, R., 2017, April. Word sense disambiguation: A unified evaluation framework and empirical comparison. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers (pp. 99-110).
* Raganato, A., Bovi, C.D. and Navigli, R., 2017, September. Neural sequence learning models for word sense disambiguation. In Proceedings of the 2017 conference on empirical methods in natural language processing (pp. 1156-1167).
* Vial, L., Lecouteux, B. and Schwab, D., 2018, May. UFSAC: Unification of sense annotated corpora and tools. In Language Resources and Evaluation Conference (LREC).
* Vial, L., Lecouteux, B. and Schwab, D., 2019. Sense vocabulary compression through the semantic knowledge of wordnet for neural word sense disambiguation. arXiv preprint arXiv:1905.05677.
* Yap, B.P., Koh, A. and Chng, E.S., 2020. Adapting BERT for word sense disambiguation with gloss selection objective and example sentences. arXiv preprint arXiv:2009.11795.
* Huang, L., Sun, C., Qiu, X. and Huang, X., 2019. GlossBERT: BERT for word sense disambiguation with gloss knowledge. arXiv preprint arXiv:1908.07245.
* Pasini, T. and Camacho-Collados, J., 2018. A short survey on sense-annotated corpora. arXiv preprint arXiv:1802.04744.
* Luo, F., Liu, T., He, Z., Xia, Q., Sui, Z. and Chang, B., 2018. Leveraging gloss knowledge in neural word sense disambiguation by hierarchical co-attention. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 1402-1411).
* Loureiro, D., Rezaee, K., Pilehvar, M.T. and Camacho-Collados, J., 2021. Analysis and evaluation of language models for word sense disambiguation. Computational Linguistics, 47(2), pp.387-443.
* Vandenbussche, P.Y., Scerri, T. and Daniel Jr, R., 2021, January. Word sense disambiguation with Transformer models. In Proceedings of the 6th Workshop on Semantic Deep Learning (SemDeep-6) (pp. 7-12).
