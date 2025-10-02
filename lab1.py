import pandas as pd

pd.set_option("display.precision", 2)
data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

# 1 вопрос
print("1 вопрос:\n")
print(f'{data['Sex'].value_counts()['male']} мужчин и {data['Sex'].value_counts()['female']} женщин')

# 2 вопрос
print("\n2 вопрос:\n")
print(data['Pclass'].value_counts().sort_index())
print(f"\nДля мужчин:\n{data[data['Sex'] == 'male']['Pclass'].value_counts().sort_index()}")
print(f"\nДля женщин:\n{data[data['Sex'] == 'female']['Pclass'].value_counts().sort_index()}")
print(f"\nКоличество людей на борту с 2 класса:\n{data[data['Pclass'] == 2].shape[0]}")

# 3 вопрос
print("\n3 вопрос:\n")
print(f"Медиана: {round(data['Fare'].median(), 2)}, \nСтандартное отклонение: {round(data['Fare'].std(), 2)}")

# 4 вопрос
print("\n4 вопрос:\n")
print("Да" if data[data['Survived'] == 1]['Age'].mean() > data[data['Survived'] == 0]['Age'].mean() else "Нет")

# 5 вопрос
print("\n5 вопрос:\n")
count_young_all = data[data['Age'] < 30].shape[0]
count_old_all = data[data['Age'] > 60].shape[0]

count_young = data[(data['Survived'] == 1) & (data['Age'] < 30)].shape[0]
count_old = data[(data['Survived'] == 1) & (data['Age'] > 60)].shape[0]
print(f"{round((count_young / count_young_all) * 100, 1)}% среди молодежи и {round((count_old / count_old_all) * 100, 1)}% среди пожилых")

# 6 вопрос
print("\n6 вопрос:\n")
count_male_all = data[data['Sex'] == 'male'].shape[0]
count_female_all = data[data['Sex'] == 'female'].shape[0]

count_male = data[(data['Survived'] == 1) & (data['Sex'] == 'male')].shape[0]
count_female = data[(data['Survived'] == 1) & (data['Sex'] == 'female')].shape[0]
print(f"{round((count_male / count_male_all) * 100, 1)}% среди мужчин и {round((count_female / count_female_all) * 100, 1)}% среди женщин")

# 7 вопрос
print("\n7 вопрос:")
first_names = data.loc[data['Sex'] == 'male', 'Name'].str.split(',').str[1].str.split().str[1]
print(first_names.value_counts().head(1))

# task 8
print("\n8. Как средний возраст мужчин / женщин зависит от Pclass? Выберите все правильные утверждения:")
means = pd.crosstab(data['Pclass'], data['Sex'], values=data['Age'], aggfunc="mean")

print("\tВ среднем мужчины 1 класса старше 40 лет")
if means.iloc[0, 1] > 40:
    print('\t\tДа')
else:
    print('\t\tНет')

print("\tВ среднем женщины 1 класса старше 40 лет")
if means.iloc[0, 0] > 40:
    print('\t\tДа')
else:
    print('\t\tНет')
print("\tМужчины всех классов в среднем старше, чем женщины того же класса")
if (means.iloc[:, 1] > means.iloc[:, 0]).all():
    print('\t\tДа')
else:
    print('\t\tНет')
print("\tВ среднем, пассажиры первого класса старше, чем пассажиры 2-го класса, которые старше, чем пассажиры 3-го класса.")
if means.iloc[0].mean() > means.iloc[1].mean() > means.iloc[2].mean():
    print('\t\tДа')
else:
    print('\t\tНет')
