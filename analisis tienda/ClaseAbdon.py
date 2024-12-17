import pandas as pd
import numpy as np

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
