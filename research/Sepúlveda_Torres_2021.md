### Exploring Summarization to Enhance Headline Stance Detection

**Summary**
- They use _extractive_ summarization, which involves cutting and pasting together segments of the original text. The other option is _abstractive_ summarization, which generates new text.
- Provides a good motivation for this task: people often just look at headlines as a source of news, and a tool that solves this problem well could automatically label headlines as possibly misleading.
- Previous SOTA is _Talos_ model which uses CNN encoder and 'Google News vectors'
- _Stance detection_ using a pre-trained RoBERTa was also successful.
- Employ a _two stage_ architecture that first classifies _relatedness_ and then only if related classifies as _agree_, _disagree_, or _discuss_. The idea is that it can only agree, disagree, or discuss if it is _already related_.
- They use _TextRank_ summarization algorithm, which tries to identify the most important sentences in the article.
- They use an array of similarity metrics as (manual) features in the classifier


**Thoughts**
- Their model does very well on 'relatedness' (99%), but not so well on 'agree' and 'disagree', (75%, 63%). This makes sense given the similarity metrics that they're using, which are likely to find surface-level matches between the two. A good target for us would be to improve on the 'agree/disagree' performance, as this is what the model is all about (iirc some of the unrelated samples are synthetic - not so realistic.). The ideal model here would construct a model of facts described in both the body and headline, and compare for inconcistencies. 