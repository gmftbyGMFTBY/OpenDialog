## Format of the processed dataset
Single-turn dialog format, the multi-turn will be processed into the single-turn mode, and the first utterance is the context (multi-turn use the [SEP] separate operator to cancatenate the multi-turn conversation context), the second utterance is the response.
```python
你好啊
你好

今天天气真不错[SEP]确实
适合放风筝
```

## Prepare datasets
1. generative data
```bash
python collect.py --mode generative
```

2. retrieval data
```bash
python collect.py --mode retrieval
```

3. When2Talk dataset
    * Ubuntu dataset
    * Dailydialog dataset cut by the `nltk`

4. utils contains the `EDA` techiques for generating the augmentation responses

## How to process the datasets
