def create_folds(ids, num_folds=5):
    '''
    Helper function for creating cross validation folds.

    Here each fold only contains the ids and not image, id pairs.
    That is because the training/validation sets used in Few Shot Learning
    cannot contain examples of the same class.
    '''
    folds = []
    for i in range(num_folds):
        start_fold = (len(ids)//num_folds)*i
        end_fold = (len(ids)//num_folds)*(i+1)

        if i == num_folds-1:
            end_fold = len(ids)

        fold = ids[start_fold:end_fold]
        folds += [fold]

    return folds
