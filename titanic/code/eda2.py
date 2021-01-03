guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j +1)]['Age'].dropna()


            age_guess = guess_df.median()

            guess_ages[i,j] = int(age_guess/ 0.5 + 0.5) *0.5

    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isn)]
