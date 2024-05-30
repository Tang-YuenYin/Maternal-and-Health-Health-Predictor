# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set(style="whitegrid")

# import warnings
# warnings.filterwarnings('ignore')

# # Load the dataset
# df = pd.read_csv("ml_model.sav")
# df.head()

# df.describe()

# df.info()

# df['fetal_health'] = df['fetal_health'].astype(int)
# df.info()

# # Get the duplicate rows
# duplicate_rows = df[df.duplicated()]

# # Calculate the number of duplicate rows
# num_duplicates = len(duplicate_rows)

# # Drop the duplicate rows from the DataFrame
# df.drop_duplicates(inplace=True)

# # Print the number of duplicate rows removed
# print("Number of duplicate rows removed:", num_duplicates)

# # Print the duplicate rows being removed
# print("Duplicate rows being removed:")
# duplicate_rows

# # Check which rows have at least one NaN value
# rows_with_nan = df.isna().any(axis=1)

# # Remove NaN values
# df = df.dropna()

# # Print the rows that have NaN values
# print("Rows with NaN being removed:")
# df[rows_with_nan]

# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# #Visualize the distribution of numerical features
# plt.figure(figsize=(25, 15))

# for i, column in enumerate(df.columns):
#     plt.subplot(5, 5, i + 1)
#     sns.histplot(data=df[column])
#     plt.title(column)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(25, 15))

# for i, column in enumerate(df.columns):
#     plt.subplot(5, 5, i + 1)
#     sns.boxplot(data=df[column])
#     plt.title(column)

# plt.tight_layout()
# plt.show()

# # List all the columns with numerical values
# df_cols = df.columns.tolist()

# #Check the skewness for each columns (data is normal if the skewness is between -2 to 2)
# for cols in df_cols:
#     skewness = df[cols].skew()
#     print("Skewness of {}: {}".format(cols, skewness))


# # Plot countplot for the fetal_health
# sns.countplot(data = df, x = "fetal_health") 

# #show the distribution of unique value in fetal health
# plt.figure(figsize=(10, 10))

# plt.pie(df['fetal_health'].value_counts(),autopct='%.2f%%',labels=["NORMAL", "SUSPECT", "PATHOLOGICAL"],colors=sns.color_palette('Blues'))

# plt.title("Class Distribution")
# plt.show()

# # Extract the fetal health and baseline heart rate values from the dataset
# fetal_health = df['fetal_health']
# baseline_heart_rate = df['baseline value']

# fig = go.Figure()

# # Add a scatter plot to visualize the relationship
# fig.add_trace(go.Scatter(x=baseline_heart_rate, y=fetal_health,
#                          mode='markers',
#                          name='Fetal Health vs Baseline Heart Rate'))

# fig.update_layout(
#     title={
#         'text': 'Relationship between Fetal Health and Baseline Value',
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     },
#     xaxis_title='Baseline Value',
#     yaxis_title='Fetal Health'
# )


# fig.show()

# fig = px.scatter(df, x='accelerations', y='fetal_movement',
#                  color='fetal_health', template='plotly_white',
#                  color_discrete_sequence=['#9AFF9A', '#EEB4B4', '#87CEFA'],
#                  title='The Impact of Accelerations and Fetal Movement on Fetal Health')

# fig.update_layout(
#     title={
#         'text': 'The Impact of Accelerations and Fetal Movement on Fetal Health',
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     }
# )

# fig.show()

# df.groupby(['uterine_contractions','fetal_health'])['fetal_health'].count()

# fig = px.histogram(df, x='uterine_contractions', color='fetal_health',
#                    template='plotly_white', barmode='group',
#                    color_discrete_sequence=px.colors.qualitative.Pastel,
#                    title='The effect of uterine contraction on fetal health')


# fig.update_layout(
#     title={
#         'text': 'The effect of uterine contraction on fetal health',
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     }
# )

# fig.show()

# fig = px.scatter_3d(df, x='light_decelerations', y='severe_decelerations', z='prolongued_decelerations', color='fetal_health',
#                     template='plotly_white', color_continuous_scale='Viridis',
#                     title='Effect of Decelerations on Fetal Health')

# fig.update_layout(
#     title={
#         'text': 'Relationship between Fetal Health and 3 Types of Decelerations',
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     },
#     xaxis_title='3 Types of Decelerations',
#     yaxis_title='Fetal Health'
# )

# fig.show()

# # List of fetal_health categories
# health_categories = [1, 2, 3]

# # Create a list to store the line graph traces
# traces = []

# # Loop over fetal_health categories
# for category in health_categories:
#     # Filter the data for the current health category
#     category_counts = df[df['fetal_health'] == category]['abnormal_short_term_variability'].value_counts().sort_index()

#     # Create a line graph trace for the current category
#     trace = go.Scatter(x=category_counts.index, y=category_counts.values, mode='lines+markers', name=f'Category {category}')
#     traces.append(trace)

# # Create the layout for the graph
# layout = go.Layout(
#     xaxis=dict(title='Percentage of Abnormal Short-term Variability'),
#     yaxis=dict(title='Count'),
#     title=dict(
#         text='Counts of Percentage of Abnormal Short-term Variability for Fetal Health Categories',
#         x=0.5
#     ),
#     showlegend=True,
#     template='plotly_white'
# )

# # Create the figure and add the line graph traces
# fig = go.Figure(data=traces, layout=layout)

# # Display the plot
# fig.show()

# displot = sns.displot(data=df, x='mean_value_of_short_term_variability', hue='fetal_health',
#                       palette='flare', kind='kde', fill=True, height=5.5, aspect=2.0)

# # Set the title at the center
# displot.ax.set_title('\nDistribution of mean value of short term variability', loc='center', y=1.0)

# plt.show()

# displot = sns.displot(data=df, x='mean_value_of_long_term_variability', hue='fetal_health',
#                       palette='flare', kind='kde', fill=True, height=5.5, aspect=2.0)

# # Set the title at the center
# displot.ax.set_title('\nDistribution of mean value of long term variability', loc='center', y=1.0)

# plt.show()

# # Correlation matrix used to find the correlation between the features and the target variable
# # Check the correlation coefficients to see which variables are highly correlated

# fig, ax = plt.subplots(figsize=(24, 10))
# mask = np.triu(np.ones_like(df.corr(), dtype=bool))  # Create a mask for the upper triangular portion
# sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='RdYlGn', annot=True, mask=mask, ax=ax)
# plt.title("Correlation Heatmap")
# plt.show()

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df['fetal_health'] = le.fit_transform(df['fetal_health'])
# df.head()

# X = df.drop(['fetal_health','histogram_mode','histogram_width','histogram_mean','histogram_min','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes','histogram_median','histogram_variance','histogram_tendency'],axis=1)
# y = df["fetal_health"]

# print('Number of features: ',len(X.columns))

# print(X.columns)

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# normal = MinMaxScaler()
# standard = StandardScaler()

# normalised_features = normal.fit_transform(X)
# normalised_data = pd.DataFrame(normalised_features, index = X.index, columns = X.columns)
# standardised_features = standard.fit_transform(X)
# standardised_data = pd.DataFrame(standardised_features, index = X.index, columns = X.columns)

# # Create subplots
# fig, ax = plt.subplots(1, 3, figsize=(30, 15))

# # Set style
# sns.set(style="ticks")

# # Original
# sns.boxplot(x='variable', y='value', data=pd.melt(df[X.columns]), ax=ax[0], palette='pastel')
# ax[0].set_title('Original')
# ax[0].set_xlabel('Variable')
# ax[0].set_ylabel('Value')
# ax[0].tick_params(axis='x', labelrotation=45)

# # MinMaxScaler
# sns.boxplot(x='variable', y='value', data=pd.melt(normalised_data[X.columns]), ax=ax[1], palette='pastel')
# ax[1].set_title('MinMaxScaler')
# ax[1].set_xlabel('Variable')
# ax[1].set_ylabel('Value')
# ax[1].tick_params(axis='x', labelrotation=45)

# # StandardScaler
# sns.boxplot(x='variable', y='value', data=pd.melt(standardised_data[X.columns]), ax=ax[2], palette='pastel')
# ax[2].set_title('StandardScaler')
# ax[2].set_xlabel('Variable')
# ax[2].set_ylabel('Value')
# ax[2].tick_params(axis='x', labelrotation=45)

# # Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.3)

# # Customize the background color
# fig.set_facecolor('#f9f9f9')

# # Remove top and right spines
# for axis in ax:
#     sns.despine(ax=axis)

# # Remove unnecessary ticks
# sns.despine(bottom=True, left=True)

# # Set plot title
# plt.suptitle('Boxplots of Variables', fontsize=16)

# # Show the plot
# plt.show()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# print('Class distribution for training set before using SMOTE:')
# y_train.value_counts()

# from imblearn.over_sampling import SMOTE
# sm = SMOTE(sampling_strategy = {0:1149, 1:500, 2:500} ,random_state=42)
# X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
# print('Class distribution for training set after resampling:')
# y_train_sm.value_counts()