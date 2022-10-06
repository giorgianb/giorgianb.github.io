# Giorgian Borca-Tasciuc

## About Me
I did my undergraduate at Stony Brook University in Electrical Engineering and Computer Science. I spent an additional year at Stony Brook to finish my master's in Computer Science.

## Projects

### Multi-Stanza
My senior Computer Science Honors Thesis was developing [mStanza](https://arxiv.org/abs/2208.03094) which is available to run and use on [github](https://github.com/giorgianb/multi-stanza). It is an extension of Stanford's [Stanza](https://github.com/stanfordnlp/stanza) NLP package which allows or the parsing of natural language sentences. Multi-Stanza was designed to address a fundamental feature of natural languages which was not incorporated into the design of Stanza: *ambiguity*. Natural languages have several valid interpretations. This can occur (non-exhautively) at the part-of-speech level of the sentence, at the lemma level of the sentence, or at the dependency level of the sentence. The following is an example of an ambiguous sentence at the dependency.
Consider the sentence:
````
I saw the man with the telescope.
````
For this sentence, it is not clear whether `the man` has the telescope or `I` have the telescope. Both intepretations are plausible. With Multi-Stanza, we can obtain both intepretations:
```python
>>> import stanza
>>> stanza.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en', depparse_n_preds=2) # This sets up a neural pipeline in English. It generates two results at the dependency parsing level.
>>> docs = nlp("I saw the man with the telescope.") # Returns a list of documents, each containing an interpretation of the sentence
>>> docs[0].sentences[0].print_dependencies() # Print the dependencies of the first interpretation. The man has the telescope
('I', 2, 'nsubj')
('saw', 0, 'root')
('the', 4, 'det')
('man', 2, 'obj')
('with', 7, 'case')
('the', 7, 'det')
('telescope', 4, 'nmod')
('.', 2, 'punct')
>>> docs[1].sentences[0].print_dependencies() # Print the dependencies of the second interpretation. I have the telescope!
('I', 2, 'nsubj')
('saw', 0, 'root')
('the', 4, 'det')
('man', 2, 'obj')
('with', 7, 'case')
('the', 7, 'det')
('telescope', 2, 'obl')
('.', 2, 'punct')
```

Multi-Stanza attempts to sort sentences by their plausibility of intepretation, from most plausible to least plausible, and offers rich mechanisms to tailor how that sorting is done towards the application at hand.


### Neural Network Fairness
Working with Professor Stanley Bak, Professor Steven Skiena, and Xingzhi Guo, I developed a [framework](https://github.com/giorgianb/nn_fairness) for quantifying *provable* fairness for neural networks. *H*-Polytopes that represent the entire input set are propagated through the network, and the location of each point in the input set is tracked. This can be used to measure two key metrics developed for the framework in order to quantify fairness.

This framework can be used to measure two key measures developed for the quantification of neural network fairness. I explain their motivation, derivation, and give examples in [this document](https://github.com/giorgianb/giorgianb.github.io/blob/5c6aa4b79367afb80294b4077ac0878253f462ac/fairness-exploration.pdf). These metrics, named *advantage* and *preference*, roughly map to the legal notions of *disparate treatment* and *disparate impact*, respectively. However, they have the following important advantages:

- **Precision**: Each metric quantifies precisely the reason why unfair outcomes occur. The metrics also allow specifying *valid discrimination criteria* (i.e.: at least a high school education is required). This allows for the precise investigation and diagnostic of *proxy variables* used to illegitimately determine the person's protected class. 
- **Intepretabily**: The metrics are straightforwardly interpretable to allow for clear reasoning if they indicate an unacceptable amount of unfairness. For example, an *advantage* of 10\% means that 10\% of black individuals would have been interviewed if they had been white, all else being equal. A *preference* of 10\% could mean that, applying the same rule for both male and female candidates, 10\% more males than females are accepted. The classes depend on the problems at hand, and the metrics allow the handling of multiple classes.
<!--
The *Advantage* is used to test for differing criteria for acceptance among different classes. It gives the proportion of individuals of class `R_1` that would have been accepted by the classifier had they been of class `R_2`. If the *Advantage* is `0.1` for a model classifying whether individuals should be interviewed for a job, this would indicate that 10% of the individuals of class `R_1` would have been interviewed *if only* they had been of class `R_2`, all other things being equal. An *Advantage* of `0` means that the same evaluation criteria are applied to both class `R_1` and class `R_2`. 

The *Preference* is used to test the model's preference for features that occur more frequently in one class over the other. It gives the difference in the proportion of accepted individuals from class `R_1` and class `R_2`, *counterfactually* evaluating the individuals in class `R_2` under the criteria of class `R_1`. This last constraint is necessary to discount differences in treatment in between the two classes, which can be measured using the *Advantage*. A *Preference* of `0.1` means that `X + 10%` of individuals of class `R_1` would be accepted, and `X`% of individuals of class `R_2` would be accepted when evaluated using the rules for class `R_1`.
-->
