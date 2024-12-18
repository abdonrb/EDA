import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataFrameAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa la clase con un DataFrame
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El argumento debe ser un DataFrame de pandas.")
        self.df = dataframe

    def resumen(self) -> pd.DataFrame:
        """
        Retorna un resumen detallado del dataset en formato DataFrame:
        - Tipo de Dato
        - Cardinalidad
        - % Cardinalidad
        - Valores Faltantes
        - % Valores Faltantes
        - Categoría
        """
        total_rows = len(self.df)
        summary = []

        for col in self.df.columns:
            # Tipo de dato
            data_type = self.df[col].dtype

            # Cardinalidad y % Cardinalidad
            cardinality = self.df[col].nunique()
            cardinality_pct = (cardinality / total_rows) * 100

            # Valores faltantes y % Valores faltantes
            missing = self.df[col].isnull().sum()
            missing_pct = (missing / total_rows) * 100

            # Determinar la categoría de la columna
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if cardinality == 2:
                    category = "Binaria"
                elif np.issubdtype(self.df[col].dtype, np.integer):
                    category = "Numérica Discreta"
                else:
                    category = "Numérica Continua"
            elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                if cardinality == 2:
                    category = "Binaria"
                else:
                    category = "Categórica Nominal"
            else:
                category = "Otro"

            # Clasificar "rowid" o índices numéricos
            if "id" in col.lower() or col.lower() == "rowid":
                category = "Índice Numérico"

            # Añadir fila al resumen
            summary.append({
                "Columna": col,
                "Tipo de Dato": data_type,
                "Cardinalidad": cardinality,
                "% Cardinalidad": round(cardinality_pct, 2),
                "Valores Faltantes": missing,
                "% Valores Faltantes": round(missing_pct, 2),
                "Categoría": category
            })

        # Crear DataFrame resumen
        summary_df = pd.DataFrame(summary)
        return summary_df

    def describe_numeric(self) -> pd.DataFrame:
        """
        Análisis estadístico detallado de variables numéricas:
        - Media, mediana, moda
        - Desviación estándar
        - Cuartiles
        - Asimetría y curtosis
        """
        numeric_df = self.df.select_dtypes(include=['number'])  # Filtrar solo variables numéricas
        
        # Calcular estadísticas
        stats = numeric_df.describe().T
        stats['mean'] = numeric_df.mean()
        stats['median'] = numeric_df.median()
        stats['mode'] = numeric_df.mode().iloc[0]
        stats['std_dev'] = numeric_df.std()
        stats['skewness'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurt()
        
        return stats[['count', 'mean', 'median', 'mode', 'std_dev', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]

    def describe_categorical(self) -> pd.DataFrame:
        """
        Análisis de variables categóricas:
        - Frecuencias
        - Proporciones
        - Valores únicos
        """
        categorical_df = self.df.select_dtypes(include=['object', 'category'])  # Filtrar variables categóricas
        
        # Calcular estadísticas
        stats = {
            "unique_values": categorical_df.nunique(),
            "most_frequent": categorical_df.mode().iloc[0],
            "frequency": categorical_df.apply(lambda x: x.value_counts().iloc[0]),
            "proportion": round((categorical_df.apply(lambda x: x.value_counts(normalize=True).iloc[0])*100),2)
        }
        
        return pd.DataFrame(stats)
    
    def plot_numeric(self):
        """
        Genera histogramas y boxplots para todas las variables numéricas.
        """
        numeric_df = self.df.select_dtypes(include=['number'])
        for col in numeric_df.columns:
            plt.figure(figsize=(14, 6))
            
            # Histograma
            plt.subplot(1, 2, 1)
            sns.histplot(numeric_df[col], kde=True, bins=20, color='blue')
            plt.title(f"Distribución de {col}")
            plt.xlabel(col)
            plt.ylabel("Frecuencia")
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=numeric_df[col], color='green')
            plt.title(f"Boxplot de {col}")
            plt.xlabel(col)
            
            plt.tight_layout()
            plt.show()

    def plot_categorical(self):
        """
        Genera gráficos de barras y pie charts para todas las variables categóricas.
        """
        categorical_df = self.df.select_dtypes(include=['object', 'category'])
        for col in categorical_df.columns:
            plt.figure(figsize=(14, 6))
            
            # Gráfico de barras
            plt.subplot(1, 2, 1)
            sns.countplot(y=categorical_df[col], order=categorical_df[col].value_counts().index, color='blue')
            plt.title(f"Frecuencia de {col}")
            plt.xlabel("Frecuencia")
            plt.ylabel(col)
            
            # Pie chart
            plt.subplot(1, 2, 2)
            categorical_df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap="viridis")
            plt.title(f"Proporción de {col}")
            plt.ylabel("")
            
            plt.tight_layout()
            plt.show()    
