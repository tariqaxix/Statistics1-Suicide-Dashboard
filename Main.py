import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import json
import plotly.express as px
import numpy as np


@st.cache
def load_data():
    file_path = r"C:\Users\tariq.aziz\OneDrive - University of Central Asia\Desktop\Statistics Project\merged_suicide_rates_dataset.csv"
    data = pd.read_csv(file_path)
    return data


data = load_data()


def clean_data(data):
    threshold = 0.5 * len(data)
    columns_to_drop = data.isnull().sum(
    )[data.isnull().sum() > threshold].index
    data_cleaned = data.drop(columns=columns_to_drop)
    data_cleaned = data_cleaned.dropna()

    Q1 = data_cleaned.quantile(0.25)
    Q3 = data_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data_cleaned[~((data_cleaned < (
        Q1 - 1.5 * IQR)) | (data_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]

    columns_to_drop = ['country-year', 'year_datetime']  # Adjust as needed
    data_cleaned = data_cleaned.drop(columns=columns_to_drop)

    return data_cleaned


def mean(data):
    return sum(data) / len(data)


def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]


def standard_deviation(data):
    mu = mean(data)
    return (sum((x - mu) ** 2 for x in data) / len(data)) ** 0.5


def variance(data):
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)


st.title("Suicide Rates Analysis")

option = st.sidebar.selectbox(
    "Choose the Analysis you want to perform",
    ("View Data", "Statistical Analysis",
     "Data Visualization", "Correlation Analysis")
)

if option == "View Data":
    st.subheader("Dataset")
    st.write(data)

elif option == "Statistical Analysis":
    st.subheader("Statistical Analysis")
    selected_columns = [
        "suicides_no",
        "suicides/100k pop",
        "HDI for year",
        "total_unemployment_rate",
        "female_to_male_labor_rate",
        "male_life_expectancy_years",
        "female_life_expectancy_years"
    ]
    column = st.selectbox("Select Column for Analysis", selected_columns)

    if st.button("Calculate"):
        data_column = data[column].dropna()
        mean_value = np.mean(data_column)
        median_value = np.median(data_column)
        std_dev = np.std(data_column)
        var = np.var(data_column)

        st.write("Mean:", mean_value)
        st.write("Median:", median_value)
        st.write("Standard Deviation:", std_dev)
        st.write("Variance:", var)

elif option == "Data Visualization":
    st.subheader("Data Visualization")

    plot_type = st.selectbox("Select Plot Type", ["Box Plot", "Bar Plot", "Pie Chart", "Histogram", "Bubble Chart (Country vs Suicide/100k pop)",
                             "Altair - GDP, Suicide Number and Gender", "Plotly - World Map of Suicides", "Scatter Plot", "Dot Plot"])

    if plot_type == "Box Plot" and st.button('Show Box Plot'):
        fig, ax = plt.subplots()
        sns.boxplot(x='sex', y='suicides/100k pop', data=data, ax=ax)
        ax.set_title('Box Plot of Suicides per 100k Population by Gender')
        st.pyplot(fig)

    elif plot_type == "Bar Plot" and st.button('Show Bar Plot'):
        fig, ax = plt.subplots()
        data['age'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Data by Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    elif plot_type == "Pie Chart" and st.button('Show Pie Chart'):
        fig, ax = plt.subplots(figsize=(6, 6))
        data['sex'].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=140, ax=ax)
        ax.set_title('Distribution of Data by Gender')
        ax.set_ylabel('')
        st.pyplot(fig)

    elif plot_type == "Histogram" and st.button('Histogram'):
        fig, ax = plt.subplots()
        sns.histplot(data['gdp_per_capita ($)'], kde=True, ax=ax)
        ax.set_title('Histogram of GDP per Capita')
        ax.set_xlabel('GDP per Capita ($)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    elif plot_type == "Altair - GDP, Suicide Number and Gender" and st.button('Show Altair'):
        gdp_suicide_gender_chart = alt.Chart(data).mark_circle().encode(
            x='gdp_per_capita ($):Q',
            y='suicides_no:Q',
            color='sex:N',
            tooltip=['gdp_per_capita ($)', 'suicides_no', 'sex']
        ).interactive().properties(
            title='GDP per Capita, Suicide Number, and Gender'
        )
        st.altair_chart(gdp_suicide_gender_chart, use_container_width=True)
    elif plot_type == "Bubble Chart (Country vs Suicide/100k pop)" and st.button('Show Bubble Chart'):
        bubble_chart = alt.Chart(data).mark_circle().encode(
            alt.X('country:N', axis=alt.Axis(title='Country')),
            alt.Y('suicides/100k pop:Q',
                  axis=alt.Axis(title='Suicides per 100k Population')),
            size='suicides_no:Q',
            color='country:N',
            tooltip=['country', 'suicides/100k pop', 'suicides_no']
        ).properties(
            title='Bubble Chart: Country vs Suicides per 100k Population'
        ).interactive()
        st.altair_chart(bubble_chart, use_container_width=True)
    elif plot_type == "Plotly - World Map of Suicides" and st.button('Show Plotly'):
        fig = px.choropleth(data, locations='country',
                            locationmode='country names',
                            color='suicides_no',
                            hover_name='country',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='World Map of Suicides')
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Scatter Plot" and st.button('Show Scatter Plot'):
        scatter_chart = alt.Chart(data).mark_circle().encode(
            alt.X('female_life_expectancy_years:Q',
                  title='Female Life Expectancy (Years)'),
            alt.Y('male_life_expectancy_years:Q',
                  title='Male Life Expectancy (Years)'),
            size='female_life_expectancy_years:Q',
            color='male_life_expectancy_years:Q',
            tooltip=['country', 'female_life_expectancy_years',
                     'male_life_expectancy_years']
        ).interactive().properties(
            title='Scatter Plot of Life Expectancy by Country'
        )
        st.altair_chart(scatter_chart, use_container_width=True)
    elif plot_type == "Dot Plot" and st.button('Show Dot Plot'):
        if 'year' in data.columns and 'country' in data.columns:
            dot_chart = alt.Chart(data).mark_point().encode(
                alt.Y('country:N', title='Country', sort=None),
                alt.X('female_to_male_labor_rate:Q', title='Labor Rate'),
                tooltip=['country', 'female_to_male_labor_rate']
            ).properties(
                title='Dot Plot of Age by Country'
            )
            st.altair_chart(dot_chart, use_container_width=True)
        else:
            st.error(
                "The required columns 'age' and 'country' are not in the dataset.")

elif option == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    analysis_columns = ['suicides_no', 'population', 'suicides/100k pop',
                        'gdp_per_capita ($)', 'total_unemployment_rate']

    if all(col in data.columns for col in analysis_columns):
        if st.button("Generate Heatmap and PairPlot"):
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.heatmap(data[analysis_columns].corr(),
                        annot=True, fmt='.2f', cmap='coolwarm', ax=ax1)
            ax1.set_title('Heatmap of Correlation Between Selected Variables')
            st.pyplot(fig1)
            st.subheader("PairPlot Analysis")
            fig2 = sns.pairplot(data[analysis_columns])
            st.pyplot(fig2)
    else:
        st.error("One or more specified columns are not in the dataset.")
