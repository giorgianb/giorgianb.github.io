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

This framework can be used to measure two key measures developed
