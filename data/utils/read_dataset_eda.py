def read_dataset_eda(dataset_name):
    path = f'{dataset_name}/train.txt'
    with open(path) as f:
        data = f.read().split('\n\n')
        dialogs = [i.split('\n') for i in data if i.strip()]
        # the dialogs must only contains the context and response, only have 2 items
        for i in dialogs:
            if len(i) != 2:
                raise Exception(f'[!] the dataset must contains 2 items, but get {len(i)}')
        contexts, responses = [i[0] for i in dialogs], [i[1] for i in dialogs]
    print(f'[!] read dataset {dataset_name} over, obtain {len(dialogs)} dialogs')
    return contexts, responses
