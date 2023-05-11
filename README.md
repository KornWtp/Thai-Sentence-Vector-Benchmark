# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation on Thai STS-B and Transfer (text classification) datasets.

# Motivation
Sentence representation plays a crucial role in NLP downstream tasks such as NLI, text classification, and STS. Recent sentence representation training techniques require NLI or STS datasets.  However, there are no equivalent Thai NLI or STS datasets for sentence representation training.
To address this problem, we train a sentence representation model with an unsupervised technique called SimCSE.

We show that it is possible to train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the Google Colab (V100) where the performance of [SimCSE-XLM-R](https://huggingface.co/mrp/simcse-model-roberta-base-thai) is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. We evaluate the Spearman correlation score of the sentence representations’ performance on Thai STS-B (translated version of [STS-B](https://github.com/facebookresearch/SentEval)). In addition, we evalute the accuracy and F1 scores of Thai text classification datasets [HERE](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb).

# How do we train unsupervised sentence representation?
- We use [SimCSE:Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) on multilingual LM models (mBERT, distil-mBERT, XLM-R) and a monolingual model (WangchanBERTa).
- Training data: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ).
- Example: [SimCSE-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb).
- Training Example on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb
## Why SimCSE?
- Easy to train
- Compatible with every model
- Does not require any annotated dataset
- The performance of XLM-R (unsupervised) and m-Distil-BERT (supervised and trained on > 1B sentences) are similar (1% difference in correlation).

# What about Supervised Learning?
- We recommend [sentence-bert](https://github.com/UKPLab/sentence-transformers), which you can train with NLI, STS, triplet loss, contrastive loss, etc.

# Multilingual Representation?
- My new work => CL-ReLKT: https://github.com/mrpeerat/CL-ReLKT (NAACL'22)

# Thai semantic textual similarity benchmark
- We use [STS-B translated ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate.
- How to evaluate sentence representation: [SentEval.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/SentEval.ipynb) 
- For the easy-to-implement version: [Easy_Evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/Easy_Evaluation.ipynb)
- How to evaluate sentence representation on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Base Models</th>
            <th>Spearman's Correlation (*100)</th>
            <th>Supervised?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">66.43</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">70.65</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=9>>100M</td>
            <td><a href="https://huggingface.co/mrp/simcse-model-distil-m-bert">simcse-model-distil-m-bert</a></td>
            <td align="center">38.84</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-m-bert-thai-cased">simcse-model-m-bert-thai-cased</a></td>
            <td align="center">39.26 </td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-wangchanberta">simcse-model-wangchanberta</a></td>
            <td align="center">52.66</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-roberta-base-thai">simcse-model-roberta-base-thai</a></td>
            <td align="center">62.60</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2">distiluse-base-multilingual-cased-v2</a></td>
            <td align="center">63.50</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2">paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">80.11</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">66.21</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">76.56</td>
            <td></td>
        </tr>
    </tbody>
</table>


# Thai transfer benchmark
- We use [Wisesight](https://huggingface.co/datasets/wisesight_sentiment), [Wongnai](https://huggingface.co/datasets/wongnai_reviews), and [Generated review](https://huggingface.co/datasets/generated_reviews_enth) datasets.
- How to evaluate: [Transfer_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb)

## Wisesight
<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th> Base Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
            <th>Supervised?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">61.55</td>
            <td align="center">62.19</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">64.77</td>
            <td align="center">65.30</td>
            <td></td>
        </tr>
         <tr>
            <td rowspan=9>>100M</td>
            <td><a href="https://huggingface.co/mrp/simcse-model-distil-m-bert">simcse-model-distil-m-bert</a></td>
            <td align="center">55.37</td>
             <td align="center">55.92</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-m-bert-thai-cased">simcse-model-m-bert-thai-cased</a></td>
            <td align="center">56.72</td> 
            <td align="center">56.95</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-wangchanberta">simcse-model-wangchanberta</a></td>
            <td align="center">62.22</td>
            <td align="center">63.06</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-roberta-base-thai">simcse-model-roberta-base-thai</a></td>
            <td align="center">61.96</td>
            <td align="center">62.48</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2">distiluse-base-multilingual-cased-v2</a></td>
            <td align="center">63.31</td>
            <td align="center">63.74</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2">paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">67.05</td>
            <td align="center">67.67</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">65.07</td>
            <td align="center">65.28</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">67.84</td>
            <td align="center">68.31</td>
            <td></td>
        </tr>
    </tbody>
</table>


## Wongnai
<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th> Base Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
            <th>Supervised?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">42.67</td>
            <td align="center">44.78</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">43.38</td>
            <td align="center">45.99</td>
            <td></td>
        </tr>
         <tr>
            <td rowspan=9>>100M</td>
            <td><a href="https://huggingface.co/mrp/simcse-model-distil-m-bert">simcse-model-distil-m-bert</a></td>
            <td align="center">36.56</td>
             <td align="center">37.31</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-m-bert-thai-cased">simcse-model-m-bert-thai-cased</a></td>
            <td align="center">39.63</td> 
            <td align="center">38.96</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-wangchanberta">simcse-model-wangchanberta</a></td>
            <td align="center">41.38</td>
            <td align="center">37.46</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-roberta-base-thai">simcse-model-roberta-base-thai</a></td>
            <td align="center">44.11</td>
            <td align="center">40.34</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2">distiluse-base-multilingual-cased-v2</a></td>
            <td align="center">37.76</td>
            <td align="center">40.07</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2">paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">45.20</td>
            <td align="center">46.72</td>
            <td align="center">&#10004;</td>
        </tr>
         <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">41.32</td>
            <td align="center">41.57</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">47.22</td>
            <td align="center">48.63</td>
            <td></td>
        </tr>
    </tbody>
</table>
## Generated Review


<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th> Base Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
            <th>Supervised?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">54.26</td>
            <td align="center">52.69</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">58.22</td>
            <td align="center">57.03</td>
            <td></td>
        </tr>
         <tr>
            <td rowspan=9>>100M</td>
            <td><a href="https://huggingface.co/mrp/simcse-model-distil-m-bert">simcse-model-distil-m-bert</a></td>
            <td align="center">38.29 </td>
             <td align="center">37.10</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-m-bert-thai-cased">simcse-model-m-bert-thai-cased</a></td>
            <td align="center">38.30</td> 
            <td align="center">36.63</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-wangchanberta">simcse-model-wangchanberta</a></td>
            <td align="center">46.63</td>
            <td align="center">42.60</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/mrp/simcse-model-roberta-base-thai">simcse-model-roberta-base-thai</a></td>
            <td align="center">42.93</td>
            <td align="center">42.81</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2">distiluse-base-multilingual-cased-v2</a></td>
            <td align="center">50.62</td>
            <td align="center">48.90</td>
            <td align="center">&#10004;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2">paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">57.48</td>
            <td align="center">56.35</td>
            <td align="center">&#10004;</td>
        </tr>
         <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">49.81</td>
            <td align="center">47.94</td>
            <td></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">58.00</td>
            <td align="center">56.80</td>
            <td></td>
        </tr>
    </tbody>
</table>


# Thank you many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)

Acknowledgments:
- Can: proofread
- Charin: proofread + idea

![1_3JJRwT1f2zTK1hx36-qXdg (1)](https://user-images.githubusercontent.com/21156980/139905794-5ce1389f-63e4-4da0-83b8-5b1aa3983222.jpeg)
