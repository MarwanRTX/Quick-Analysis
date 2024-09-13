import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix, parallel_coordinates
from matplotlib.backends.backend_pdf import PdfPages
import missingno as msno
import matplotlib
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def remove_outliers(self):
        try:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            data_numeric = self.data.select_dtypes(include=numerics)
            
            for col in data_numeric.columns:
                Q1 = data_numeric[col].quantile(0.25)
                Q3 = data_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = 1.5
                self.data = self.data[~((self.data[col] < Q1 - threshold * IQR) | (self.data[col] > Q3 + threshold * IQR))]
            
            return self.data
        except Exception as e:
            print(f"Error in removing outliers: {e}")

    def remove_pattern(self, column, pattern):
        try:
            if column in self.data.columns:
                self.data[column] = self.data[column].astype(str).replace(pattern, '', regex=True)
            return self.data
        except Exception as e:
            print(f"An error occurred while removing pattern: {e}")

    def detect_outliers(self):
        try:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            data_numeric = self.data.select_dtypes(include=numerics)
            outlier_rows = []

            for col in data_numeric.columns:
                Q1 = data_numeric[col].quantile(0.25)
                Q3 = data_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = 1.5
                outliers = data_numeric[(data_numeric[col] < Q1 - threshold * IQR) | (data_numeric[col] > Q3 + threshold * IQR)]
                if not outliers.empty:
                    outlier_rows.append(outliers)
            
            if outlier_rows:
                combined_outliers = pd.concat(outlier_rows)
                return combined_outliers.sample(5) if len(combined_outliers) > 0 else pd.DataFrame()
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while detecting outliers: {e}")

    def generate_plots(self, filename):
        try:
            downloads_folder = 'app/static/downloads'
            file_base_name = os.path.splitext(filename)[0]
            file_folder = os.path.join(downloads_folder, file_base_name)
            individual_pdfs_folder = os.path.join(file_folder, 'plots')

            os.makedirs(individual_pdfs_folder, exist_ok=True)

            combined_filepath = os.path.join(individual_pdfs_folder, f"{file_base_name}_combined.pdf")

            with PdfPages(combined_filepath) as combined_pdf:
                plot_methods = [
                    self._generate_histograms,
                    self._generate_box_plots,
                    self._generate_scatter_matrix,
                    self._generate_correlation_heatmap,
                    self._generate_line_plots,
                    self._generate_bar_plots,
                    self._generate_violin_plots,
                    self._generate_density_plots,
                    self._generate_parallel_coordinates,
                    self._generate_pca_plot,
                ]

                # Check if there are any numeric or datetime columns before proceeding
                numeric_data = self.data.select_dtypes(include=['number', 'datetime'])
                if numeric_data.empty:
                    print("No numeric or datetime columns found, skipping all numeric-based plots.")
                    print("File uploaded successfully!")  # Ensure success message is always printed
                    return

                for plot_method in plot_methods:
                    plot_name = plot_method.__name__.replace('_generate_', '').replace('_', ' ').title()
                    plot_filename = os.path.join(individual_pdfs_folder, f"{plot_name}.pdf")
                    
                    try:
                        with PdfPages(plot_filename) as pdf:
                            self.data.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

                            plot_method(pdf, combined_pdf)
                        print(f"{plot_name} generated and saved as {plot_filename}.")
                    except Exception as e:
                        print(f"Failed to generate {plot_name}: {e}")

                print(f"All individual plots and the combined PDF saved to {individual_pdfs_folder}")

        except Exception as e:
            print(f"An error occurred while generating plots: {e}")

        # Ensure the success message is always printed
        print("File uploaded successfully!")


    def _generate_histograms(self, pdf, combined_pdf):
        try:
            self.data.hist(figsize=(10, 10))
            plt.suptitle('Histograms')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating histograms: {e}")

    def _generate_box_plots(self, pdf, combined_pdf):
        try:
            self.data.plot(kind='box', subplots=True, layout=(int(len(self.data.columns) / 3) + 1, 3), figsize=(12, 8))
            plt.suptitle('Box Plots')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating box plots: {e}")

    def _generate_scatter_matrix(self, pdf, combined_pdf):
        try:
            pca = PCA(n_components=min(self.data.select_dtypes(include=['number']).shape))
            reduced_data = pca.fit_transform(self.data.select_dtypes(include=['number']))
            scatter_matrix(pd.DataFrame(reduced_data), figsize=(12, 12), diagonal='kde')
            plt.suptitle('Scatter Plot Matrix')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating scatter matrix: {e}")

    def _generate_correlation_heatmap(self, pdf, combined_pdf):
        try:
            numeric_data = self.data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
                plt.title('Correlation Heatmap')
                pdf.savefig()
                combined_pdf.savefig()
                plt.close()
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

    def _generate_line_plots(self, pdf, combined_pdf, downsample_factor=10):
        try:
            if len(self.data) > downsample_factor * 100:
                sampled_data = self.data.iloc[::downsample_factor, :]  # Downsample every nth row
            else:
                sampled_data = self.data

            print(f"Generating line plots with {len(sampled_data)} rows.")
            
            sampled_data.plot(subplots=True, layout=(int(len(sampled_data.columns) / 3) + 1, 3), figsize=(12, 8), kind='line')
            plt.suptitle('Line Plots')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating line plots: {e}")


    def _generate_bar_plots(self, pdf, combined_pdf):
        try:
            for column in self.data.select_dtypes(include=['object', 'category']):
                self.data[column].value_counts().plot(kind='bar', figsize=(10, 6))
                plt.title(f'Bar Plot of {column}')
                pdf.savefig()
                combined_pdf.savefig()
                plt.close()
        except Exception as e:
            print(f"Error generating bar plots: {e}")

    def _generate_violin_plots(self, pdf, combined_pdf):
        try:
            for column in self.data.select_dtypes(include=['number']):
                sns.violinplot(x=column, data=self.data, inner="quart")
                plt.title(f'Violin Plot of {column}')
                pdf.savefig()
                combined_pdf.savefig()
                plt.close()
        except Exception as e:
            print(f"Error generating violin plots: {e}")

    def _generate_density_plots(self, pdf, combined_pdf):
        try:
            self.data.plot(kind='density', subplots=True, layout=(int(len(self.data.columns) / 3) + 1, 3), figsize=(12, 8))
            plt.suptitle('Density Plots')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating density plots: {e}")

    def _generate_parallel_coordinates(self, pdf, combined_pdf):
        try:
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                numeric_cols = self.data.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    parallel_coordinates(self.data[numeric_cols.insert(0, categorical_cols[0])], categorical_cols[0])
                    plt.title('Parallel Coordinates Plot')
                    pdf.savefig()
                    combined_pdf.savefig()
                    plt.close()
        except Exception as e:
            print(f"Error generating parallel coordinates plot: {e}")

    def _generate_pca_plot(self, pdf, combined_pdf):
        try:
            numeric_data = self.data.select_dtypes(include=['number'])
            if numeric_data.empty or numeric_data.shape[1] < 2:
                print("Not enough numeric data for PCA.")
                return

            imputer = SimpleImputer(strategy='mean')
            numeric_data_imputed = imputer.fit_transform(numeric_data)

            pca = PCA(n_components=2)
            components = pca.fit_transform(numeric_data_imputed)

            plt.scatter(components[:, 0], components[:, 1], c='b', marker='o')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA Plot')
            pdf.savefig()
            combined_pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating PCA plot: {e}")


    def replace_with_mean_or_median(self, column, method='mean'):
        try:
            if method == 'mean':
                value = self.data[column].mean()
            elif method == 'median':
                value = self.data[column].median()
            else:
                raise ValueError("Method must be 'mean' or 'median'")
            
            self.data[column].replace(np.nan, value, inplace=True)
            return self.data
        except Exception as e:
            print(f"Error replacing values with {method}: {e}")

    def plot_null_patterns(self, filename):
        try:
            downloads_folder = 'app/static/downloads'
            file_base_name = os.path.splitext(filename)[0]
            file_folder = os.path.join(downloads_folder, file_base_name)
            individual_images_folder = os.path.join(file_folder, 'plots')

            os.makedirs(individual_images_folder, exist_ok=True)

            seaborn_path = os.path.join(individual_images_folder, f"{file_base_name}_seaborn_null.png")
            missingno_path = os.path.join(individual_images_folder, f"{file_base_name}_missingno_null.png")

            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Seaborn Heatmap of Null Values')
            plt.savefig(seaborn_path)
            plt.close()
            print(f"Seaborn heatmap saved to {seaborn_path}")

            plt.figure(figsize=(12, 8))
            msno.heatmap(self.data)
            plt.title('Missingno Heatmap of Null Values')
            plt.savefig(missingno_path)
            plt.close()
            print(f"Missingno heatmap saved to {missingno_path}")

        except Exception as e:
            print(f"Error plotting null patterns: {e}")

    def drop_nulls(self):
        try:
            df_cleaned = self.data.dropna()
            for col in df_cleaned.select_dtypes(include=['int', 'float']).columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='float')
            return df_cleaned
        except Exception as e:
            print(f"Error dropping nulls: {e}")
            return self.data

    def get_columns(self):
        if self.data is None:
            return None
        return self.data.columns
