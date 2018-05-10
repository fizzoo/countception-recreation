# countception-recreation
This repo is a translation of the core part of the original theano code at 
[https://github.com/ieee8023/NeuralNetwork-Examples/blob/723989012f3dcf437cede6db40e58deac946edea/theano/counting/count-ception.ipynb] over to keras.

It was created as the first step in adapting the strategy to a different domain (see [https://github.com/timsl/sea-lion-kaggle]), ~~and so we did not train long enough to verify the results in the original paper~~ (not to mention that there is now a newer version of the paper).
It also seems that further work has been done to the theano code since the creation of this, so some parts may be outdated.

Update: With some fiddling of the hyperparameters it is possible to get results that are very close to the original, so the code seems fine. The gaussian-shaped labels in the newer version of the paper is not implemented, only the square shaped labels.
