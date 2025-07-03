import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv('../../data/raw/UpdatedResumeDataSet.csv')

print("Initial Dataset Info:")
print(df.info())
print("\nInitial Category Distribution:")
print(df['Category'].value_counts())

plt.figure(figsize=(14,8))
sns.countplot(data=df, x='Category', order=df['Category'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Initial: Number of Resumes per Category')
plt.xlabel('Job Category')
plt.ylabel('Count')
plt.tight_layout(pad=2)
plt.show()

df['resume_len_words'] = df['Resume'].apply(lambda x: len(str(x).split()))
df['resume_len_chars'] = df['Resume'].apply(lambda x: len(str(x)))

plt.figure(figsize=(12,6))
plt.hist(df['resume_len_words'], bins=30, color='skyblue', edgecolor='black')
plt.title('Initial: Distribution of Resume Length (Words)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Resumes')
plt.tight_layout(pad=2)
plt.show()

it_categories = [
    'Java Developer', 'Testing', 'DevOps Engineer', 'Python Developer', 'Web Designing',
    'Hadoop', 'Blockchain', 'ETL Developer', 'Data Science', 'Database',
    'DotNet Developer', 'Automation Testing', 'Network Security Engineer', 'SAP Developer',
    'PMO', 'Business Analyst'
]

df_it = df[df['Category'].isin(it_categories)].reset_index(drop=True)

print("\nAfter Cleanup: IT-Related Dataset Info:")
print(df_it.info())
print("\nAfter Cleanup: IT Category Distribution:")
print(df_it['Category'].value_counts())

plt.figure(figsize=(14,8))
sns.countplot(data=df_it, x='Category', order=df_it['Category'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Final: Number of IT-Related Resumes per Category')
plt.xlabel('Job Category')
plt.ylabel('Count')
plt.tight_layout(pad=2)
plt.show()

plt.figure(figsize=(12,6))
plt.hist(df['resume_len_words'], bins=30, color='skyblue', edgecolor='black')
plt.title('Final: Distribution of Resume Length (Words) (IT Only)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Resumes')
plt.tight_layout(pad=2)
plt.show()

df_it.to_csv('it_resumes_cleaned.csv', index=False)
print("\nSaved cleaned IT resumes to 'it_resumes_cleaned.csv'")
