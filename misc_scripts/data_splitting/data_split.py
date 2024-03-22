from sklearn.model_selection import train_test_split 

with open('UIDs.txt', 'r') as f:
    UIDs = f.readlines()
    # Remove the first line as it is the comment
    UIDs = UIDs[1:]
    # Remove the newline character from each line
    UIDs = [UID.strip() for UID in UIDs]

train, test = train_test_split(UIDs, random_state=42, test_size=0.2, shuffle=False, stratify=None)

with open('train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open('test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)