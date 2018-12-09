def printout(dataset, datasplit, error, pred):
    length = 0
    with open('%s/%s.cor' % (dataset, datasplit)) as f:
        sentences = f.readlines()
        origin = [];
        length = int(len(sentences) / 3)
        for i in range(length):
            data = [sentences[i*3].strip(),sentences[i*3 + 1].strip(),sentences[i*3 + 2].strip()]
            origin.append(data)
    with open('%s/%s.text' % ('result', 'error'), 'w+') as e:
        for i in range(len(error)):
            e.write('sentence: %s\n' % origin[error[i]][0])
            e.write('aspect: %s\n' % origin[error[i]][1])
            e.write('polarity: %s\tprediction: %s\n\n' % (origin[error[i]][2], pred[i]))
