# Creamos el archivo de la APP en el interprete principal (Phyton)

#############################IMPLEMENTACIÓN DE DASHBOARD################################

# Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from funpymodeling.exploratory import freq_tbl 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

#################################################################

# Definimos la instancia de streamlit
@st.cache_resource

#################################################################

# Creamos la función de carga de datos
def load_data():
    # Lectura del archivo csv sin indice
    df1=pd.read_csv("Sydney.csv")
    

#################################################################

    # Etapa de procesamiento de Datos
   
    # ANÁLISIS UNIVARIADO DE FRECUENCIAS
    # Obtengo un análisis univariado de una variable categórica en específico
    table= freq_tbl(df1['property_type'])
    Filtro= table[table['frequency']>1000]
    tipo_de_propiedad= Filtro.set_index('property_type')   

    # Análisis univariado de otra variable categórica
    table2= freq_tbl(df1['host_response_time'])
    Filtro2= table2[table2['frequency']>100]
    tiempo_respuesta_host= Filtro2.set_index('host_response_time')     
    
    # Análisis univariado de otra variable categórica
    table3= freq_tbl(df1['room_type'])
    Filtro3= table3[table3['frequency']>100]
    tipo_de_cuarto= Filtro3.set_index('room_type')    

    # Análisis univariado de otra variable categórica
    table4= freq_tbl(df1['host_is_superhost'])
    Filtro4= table4[table4['frequency']>100]
    host_es_superhost= Filtro4.set_index('host_is_superhost') 

    # Análisis univariado de otra variable categórica
    table5= freq_tbl(df1['host_neighbourhood'])
    Filtro5= table5[table5['frequency']>1000]
    vecindario_host= Filtro5.set_index('host_neighbourhood')    

    # Análisis univariado de otra variable categórica
    table6= freq_tbl(df1['host_has_profile_pic'])
    Filtro6= table6[table6['frequency']>100]
    foto_perfil_host= Filtro6.set_index('host_has_profile_pic')    

    # Análisis univariado de otra variable categórica
    table7= freq_tbl(df1['host_identity_verified'])
    Filtro7= table7[table7['frequency']>100]
    identidad_verificada= Filtro7.set_index('host_identity_verified') 

    # Análisis univariado de otra variable categórica
    table8= freq_tbl(df1['neighbourhood'])
    Filtro8= table8[table8['frequency']>1]
    vecindario= Filtro8.set_index('neighbourhood') 
    
    # Análisis univariado de otra variable categórica
    table9= freq_tbl(df1['neighbourhood_cleansed'])
    Filtro9= table9[table9['frequency']>800]
    vecindario_limpio= Filtro9.set_index('neighbourhood_cleansed')

    # Análisis univariado de otra variable categórica
    table10= freq_tbl(df1['name'])
    Filtro10= table10[table10['frequency']>5]
    nombre= Filtro10.set_index('name')

    # Análisis univariado de otra variable categórica
    table11= freq_tbl(df1['host_name'])
    Filtro11= table11[table11['frequency']>140]
    nombre_host= Filtro11.set_index('host_name')

    # Selecciono las columnas tipo numéricas del dataframe tipo_de_propiedad
    numeric_df1 = tipo_de_propiedad.select_dtypes(['float','int'])  
    numeric_cols1= numeric_df1.columns  

    # Filtramos las variables numéricas directamente del DataFrame original df1
    # Excluimos las columnas 'Unnamed: 0' y 'host_acceptance_rate.1' si existen
    numeric_columns_df1 = df1.select_dtypes(include=['float', 'int']).columns
    numeric_columns_df1 = numeric_columns_df1.drop(['Unnamed: 0', 'host_acceptance_rate.1'], errors='ignore')

    return tipo_de_propiedad, df1, numeric_df1, numeric_cols1, tiempo_respuesta_host, tipo_de_cuarto, host_es_superhost,numeric_columns_df1, vecindario_host, foto_perfil_host,identidad_verificada,vecindario_limpio,vecindario,nombre,nombre_host

#################################################################

# Cargo los datos obtenidos de la función "load_data"
tipo_de_propiedad, df1, numeric_df1, numeric_cols1, tiempo_respuesta_host, tipo_de_cuarto, host_es_superhost, numeric_columns_df1, vecindario_host, foto_perfil_host,identidad_verificada,vecindario_limpio,vecindario,nombre,nombre_host= load_data()

######################CREACIÓN DEL DASHBOARD################################

# 1. CREACIÓN DE LA SIDEBAR
st.sidebar.title("DASHBOARD")
st.sidebar.header("Sidebar")
st.sidebar.subheader("Panel de selección")

# Agregar la imagen a la sidebar
image = "https://www.newsvoir.com/images/user/logo/_airbnb-official-logo.png"
st.sidebar.image(image, use_column_width=True)

# 2. CREACIÓN DE LOS FRAMES
Frames = st.selectbox(label= "Frames", options= ["Análisis Univariado", "Análisis de Dispersión", "Box plot", "Regresión Lineal Múltiple","Regresión No Lineal"])

#################################################################

# 1. Implementación del estilo para el Dashboard
def set_custom_style():
    st.markdown(
        """
        <style>
        /* Fondo del cuerpo principal del dashboard */
        .stApp {
            background-color: #ff595f; /* Color de fondo del cuerpo */
        }

        /* Fondo de la barra lateral */
        .css-1aumxhk {
            background-color: #001f3f !important; /* Fondo azul marino oscuro */
        }

        /* Texto y encabezados de la barra lateral */
        .stSidebar * {
            color: #000000 !important; /* Cambiar a negro para todos los textos en la barra lateral */
        }

        /* Estilo de los botones */
        .stButton>button {
            background-color: #0074D9; /* Botón azul */
            color: #ffffff; /* Texto blanco en los botones */
            border-radius: 8px;
            border: 2px solid #0074D9;
        }
        .stButton>button:hover {
            background-color: #005f99; /* Oscurecimiento al hacer hover */
        }

        /* Color de los headers del dashboard */
        h1, h2, h3, h4 {
            color: #ffffff; /* Color blanco para los headers */
            text-align: center; /* Centramos los títulos */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Llamada a los estilos personalizados
set_custom_style()


# 3. CONTENIDO DEL FRAME 1
if Frames == "Análisis Univariado":
    st.title("SYDNEY 🌆")
    st.header("Análisis Univariado de variables categóricas")
    st.subheader("Bar Plot 📈")
   
    # Checkbox para mostrar dataset
    check_box = st.sidebar.checkbox(label= "Mostrar Dataset")
    if check_box:
        st.write(df1)
    
    # Selección de la variable categórica para graficar
    Variables = st.sidebar.selectbox(label= "Variable categórica", 
                                     options= ["Tipo de propiedad", 
                                               "Tiempo de respuesta del host", 
                                               "Tipo de cuarto", 
                                               "Host es superhost","Vecindario Host","Foto perfil host",
                                               "Identidad verificada","Vecindarios Limpios",
                                               "Nombre Air bnb","Nombre Host"])
    
    # Selección de la métrica numérica para graficar
    Vars_Num = st.selectbox(label= "Métrica a graficar", 
                            options= ["frequency", "percentage", "cumulative_perc"])
    
    # Configuración de colores para las gráficas
    bar_color = "#00A19D"  # Color para las barras
    pie_colors = ['#ff595f', '#0074D9', '#FF851B', '#FFDC00', '#2ECC40']  # Colores para la gráfica de pastel

    # Dependiendo de la selección en el selectbox de Variables, mostramos una gráfica
    if Variables == "Tipo de propiedad":
        figure = px.bar(data_frame=tipo_de_propiedad, 
                        x=tipo_de_propiedad.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Tipo de Propiedad",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=tipo_de_propiedad, 
                            names=tipo_de_propiedad.index, 
                            values=Vars_Num, 
                            title="Distribución - Tipo de Propiedad",
                            color_discrete_sequence=pie_colors)
    
    elif Variables == "Tiempo de respuesta del host":
        figure = px.bar(data_frame=tiempo_respuesta_host, 
                        x=tiempo_respuesta_host.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Tiempo de Respuesta del Host",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=tiempo_respuesta_host, 
                            names=tiempo_respuesta_host.index, 
                            values=Vars_Num, 
                            title="Distribución - Tiempo de Respuesta del Host",
                            color_discrete_sequence=pie_colors)
    
    elif Variables == "Tipo de cuarto":
        figure = px.bar(data_frame=tipo_de_cuarto, 
                        x=tipo_de_cuarto.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Tipo de Cuarto",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=tipo_de_cuarto, 
                            names=tipo_de_cuarto.index, 
                            values=Vars_Num, 
                            title="Distribución - Tipo de Cuarto",
                            color_discrete_sequence=pie_colors)
    
    elif Variables == "Host es superhost":
        figure = px.bar(data_frame=host_es_superhost, 
                        x=host_es_superhost.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Host es Superhost",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=host_es_superhost, 
                            names=host_es_superhost.index, 
                            values=Vars_Num, 
                            title="Distribución - Host es Superhost",
                            color_discrete_sequence=pie_colors)
    
    elif Variables == "Vecindario Host":
        figure = px.bar(data_frame=vecindario_host, 
                        x=vecindario_host.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Vecindario del Host",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=vecindario_host, 
                            names=vecindario_host.index, 
                            values=Vars_Num, 
                            title="Distribución - Vecindario del Host",
                            color_discrete_sequence=pie_colors)

    elif Variables == "Foto perfil host":
        figure = px.bar(data_frame=foto_perfil_host, 
                        x=foto_perfil_host.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Foto de perfil del host",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=foto_perfil_host, 
                            names=foto_perfil_host.index, 
                            values=Vars_Num, 
                            title="Distribución - Foto de perfil del host",
                            color_discrete_sequence=pie_colors)

    elif Variables == "Identidad verificada":
        figure = px.bar(data_frame=identidad_verificada, 
                        x=identidad_verificada.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Identidad Verificada del Host",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=identidad_verificada, 
                            names=identidad_verificada.index, 
                            values=Vars_Num, 
                            title="Distribución - Identidad Verificada del Host",
                            color_discrete_sequence=pie_colors)

    elif Variables == "Vecindarios Limpios":
        figure = px.bar(data_frame=vecindario_limpio, 
                        x=vecindario_limpio.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Vecindarios Limpios",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=vecindario_limpio, 
                            names=vecindario_limpio.index, 
                            values=Vars_Num, 
                            title="Distribución - Vecindarios Limpios",
                            color_discrete_sequence=pie_colors)

    elif Variables == "Nombre Air bnb":
        figure = px.bar(data_frame=nombre, 
                        x=nombre.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Nombre Air bnb",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=nombre, 
                            names=nombre.index, 
                            values=Vars_Num, 
                            title="Distribución - Nombre Air bnb",
                            color_discrete_sequence=pie_colors)

    elif Variables == "Nombre Host":
        figure = px.bar(data_frame=nombre_host, 
                        x=nombre_host.index, 
                        y=Vars_Num, 
                        title="Análisis Univariado - Nombre de Host",
                        color_discrete_sequence=[bar_color])
        
        # Gráfica de pastel
        figure_pie = px.pie(data_frame=nombre_host, 
                            names=nombre_host.index, 
                            values=Vars_Num, 
                            title="Distribución - Nombre de Host",
                            color_discrete_sequence=pie_colors)

    # Mostramos la gráfica seleccionada
    st.plotly_chart(figure, use_container_width=True)
    
    # Mostramos la gráfica de pastel
    st.plotly_chart(figure_pie, use_container_width=True)


# 5. CONTENIDO DEL FRAME 2
if Frames == "Análisis de Dispersión":
    # Generamos los encabezados para el dashboard
    st.markdown("<h1 style='color: white;'>SYDNEY 🌆</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Análisis de Dispersión (Scatter Plot) 📈</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Variables numéricas</h3>", unsafe_allow_html=True)

    # Mapa de calor de la correlación
    corr_matrix = df1[numeric_columns_df1].corr()
    fig_heatmap = px.imshow(corr_matrix, 
                            title="Mapa de Calor de Correlación",
                            color_continuous_scale='Viridis',
                            aspect='auto',
                            width=800,  # Ancho del gráfico
                            height=800)  # Altura del gráfico
    fig_heatmap.update_layout(title_font_color='white', font_color='white')  # Cambia el color del título y del texto
    st.plotly_chart(fig_heatmap)

    # Generamos dos cuadros de selección (x, y) para seleccionar variables numéricas a graficar
    x_selected = st.sidebar.selectbox(label="Seleccione variable para eje X", options=numeric_columns_df1)
    y_selected = st.sidebar.selectbox(label="Seleccione variable para eje Y", options=numeric_columns_df1)

    # Gráfico de dispersión (scatter plot)
    figure20 = px.scatter(data_frame=df1, x=x_selected, y=y_selected, title='Gráfico de Dispersión')
    figure20.update_layout(title_font_color='white', font_color='white')  # Cambia el color del título y del texto
    st.plotly_chart(figure20)

    # Generamos el modelo de regresión lineal
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Variables independientes (eje X) y dependientes (eje Y) seleccionadas en el dashboard
    Vars_Indep = df1[[x_selected]]
    Var_Dep = df1[y_selected]
    
    # Definimos el modelo de regresión lineal
    model = LinearRegression()
    
    # Ajustamos el modelo con las variables seleccionadas
    model.fit(X=Vars_Indep, y=Var_Dep)
    
    # Obtenemos los coeficientes y la intersección
    coeficientes = model.coef_
    intercepto = model.intercept_
    
    # Mostramos la ecuación del modelo lineal
    st.markdown("<h3 style='color: white;'>Ecuación del Modelo Lineal:</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white;'>{y_selected} = {coeficientes[0]:.4f} * {x_selected} + {intercepto:.4f}</p>", unsafe_allow_html=True)
    
    # Tabla de los coeficientes
    st.markdown("<h3 style='color: white;'>Tabla de Coeficientes:</h3>", unsafe_allow_html=True)
    coef_data = {
        'Variable': [x_selected],
        'Coeficiente': [coeficientes[0]],
        'Intercepto': [intercepto]
    }
    
    coef_df = pd.DataFrame(coef_data)
    st.markdown(coef_df.style.applymap(lambda x: 'color: white;', subset=['Variable', 'Coeficiente', 'Intercepto']).to_html(), unsafe_allow_html=True)
    
    # Mostramos la eficiencia del modelo (R^2)
    st.markdown("<h3 style='color: white;'>Eficiencia del Modelo:</h3>", unsafe_allow_html=True)
    r_squared = model.score(Vars_Indep, Var_Dep)
    st.markdown(f"<p style='color: white;'>Coeficiente de determinación (R²): {r_squared:.4f}</p>", unsafe_allow_html=True)
    
    # Predecimos los valores con el modelo ajustado
    y_pred = model.predict(X=Vars_Indep)
    
    # Insertamos las predicciones en el DataFrame original para visualización
    df1['Predicciones'] = y_pred
    
    # Graficamos las predicciones junto con los valores reales
    fig_pred = px.scatter(data_frame=df1, x=y_selected, y=y_pred, 
                          title="Comparación entre Valores Reales y Predicciones",
                          labels={'y': 'Predicciones', 'x': y_selected})
    fig_pred.update_layout(title_font_color='white', font_color='white')  # Cambia el color del título y del texto
    
    # Mostramos la gráfica comparativa
    st.plotly_chart(fig_pred)

    # Mostramos coeficiente de correlación
    coef_correl = np.sqrt(r_squared)
    st.markdown(f"<p style='color: white;'>Coeficiente de correlación: {coef_correl:.4f}</p>", unsafe_allow_html=True)





# CONTENIDO DEL FRAME 3
if Frames == "Box plot":
    st.title("SYDNEY 🌆")
    st.header("Gráficas de Caja (Box Plots) 📈")
    st.subheader("Distribución de Variables Numéricas")

    # Filtramos las variables numéricas del DataFrame original df1
    numeric_columns_df1 = df1.select_dtypes(include=['float', 'int']).columns
    numeric_columns_df1 = numeric_columns_df1.drop(['Unnamed: 0', 'host_acceptance_rate.1'], errors='ignore')

    # Selección de múltiples variables numéricas para graficar
    var_boxplot = st.sidebar.multiselect(label="Seleccione variables numéricas para Box Plot", options=numeric_columns_df1)

    # Crear un botón para actualizar el gráfico
    if st.button("Actualizar Gráfico"):
        # Graficar diagrama de caja (box plot) solo si se seleccionan variables
        if var_boxplot:
            # Reestructuramos el DataFrame para que esté en formato largo (long format) y poder graficar
            df_long = df1[var_boxplot].melt(var_name="Variable", value_name="Valor")

            # Graficar box plot con las variables seleccionadas
            fig_box = px.box(data_frame=df_long, x="Variable", y="Valor", title='Box Plot de Variables Seleccionadas')
            st.plotly_chart(fig_box)
        else:
            st.write("Por favor seleccione al menos una variable numérica para graficar.")


# CONTENIDO DEL FRAME 4
if Frames == "Regresión Lineal Múltiple":
    # Generamos los encabezados para el dashboard
    st.markdown("<h1 style='color: white;'>SYDNEY 🌆</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Regresión Lineal Múltiple 📈</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Variables numéricas</h3>", unsafe_allow_html=True)

    # Generamos el cuadro de selección para el eje Y (una sola variable dependiente)
    y_selected_multiple = st.sidebar.selectbox(label="Seleccione variable para eje Y (dependiente)", options=numeric_columns_df1)

    # Generamos el cuadro de selección múltiple para el eje X (varias variables independientes)
    x_selected_multiple = st.sidebar.multiselect(label="Seleccione variables para eje X (independientes)", options=numeric_columns_df1)

    # Aseguramos que se hayan seleccionado variables en ambos casos
    if y_selected_multiple and x_selected_multiple:
        # Variables independientes y dependientes seleccionadas
        Vars_Indep_multiple = df1[x_selected_multiple]
        Var_Dep_multiple = df1[y_selected_multiple]

        # Definimos el modelo de regresión lineal múltiple
        from sklearn.linear_model import LinearRegression
        model_multiple = LinearRegression()

        # Ajustamos el modelo con las variables seleccionadas
        model_multiple.fit(X=Vars_Indep_multiple, y=Var_Dep_multiple)

        # Obtenemos los coeficientes y la intersección del modelo
        coeficientes_multiple = model_multiple.coef_
        intercepto_multiple = model_multiple.intercept_

        # Mostramos la ecuación del modelo lineal múltiple
        st.markdown("<h3 style='color: white;'>Ecuación del Modelo Lineal Múltiple:</h3>", unsafe_allow_html=True)
        equation = f"{y_selected_multiple} = "
        for i, var in enumerate(x_selected_multiple):
            equation += f"{coeficientes_multiple[i]:.4f} * {var} + "
        equation += f"{intercepto_multiple:.4f}"
        st.markdown(f"<p style='color: white;'>{equation}</p>", unsafe_allow_html=True)

        # Tabla de los coeficientes
        st.markdown("<h3 style='color: white;'>Tabla de Coeficientes:</h3>", unsafe_allow_html=True)
        coef_data_multiple = {
            'Variable': x_selected_multiple,
            'Coeficiente': coeficientes_multiple,
            'Intercepto': [intercepto_multiple] * len(coeficientes_multiple)
        }
        coef_df_multiple = pd.DataFrame(coef_data_multiple)

        # Cambiamos el color de los coeficientes a blanco
        st.markdown(coef_df_multiple.style.applymap(lambda x: 'color: white;', subset=['Variable', 'Coeficiente', 'Intercepto']).to_html(), unsafe_allow_html=True)

        # Mostramos la eficiencia del modelo (R^2)
        st.markdown("<h3 style='color: white;'>Eficiencia del Modelo:</h3>", unsafe_allow_html=True)
        r_squared_multiple = model_multiple.score(Vars_Indep_multiple, Var_Dep_multiple)
        st.markdown(f"<p style='color: white;'>Coeficiente de determinación (R²): {r_squared_multiple:.4f}</p>", unsafe_allow_html=True)

        # Predecimos los valores con el modelo ajustado
        y_pred_multiple = model_multiple.predict(X=Vars_Indep_multiple)

        # Insertamos las predicciones en el DataFrame original para visualización
        df1['Predicciones_Multiple'] = y_pred_multiple

        # Crear un DataFrame para las variables seleccionadas y las predicciones
        df_plot = df1[x_selected_multiple].copy()
        df_plot[y_selected_multiple] = df1[y_selected_multiple]
        df_plot['Predicciones_Multiple'] = df1['Predicciones_Multiple']
        df_plot = df_plot.melt(id_vars=[y_selected_multiple, 'Predicciones_Multiple'],
                               value_vars=x_selected_multiple,
                               var_name='Variable', value_name='Valor')

        # Graficamos las predicciones junto con los valores reales
        fig_pred_multiple = px.scatter(df_plot, 
                                       x=y_selected_multiple, 
                                       y='Predicciones_Multiple',
                                       color='Variable',
                                       title="Comparación entre Valores Reales y Predicciones",
                                       labels={'Predicciones_Multiple': 'Predicciones',
                                               y_selected_multiple: 'Valores Reales'},
                                       template='plotly_dark')  # Tema oscuro

        # Mejora en la apariencia de la gráfica
        fig_pred_multiple.update_layout(
            title_font_color='white',
            font_color='white',
            xaxis_title_font_color='white',
            yaxis_title_font_color='white',
            plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            margin=dict(l=40, r=40, t=40, b=40)  # Márgenes
        )
        
        # Mostramos la gráfica comparativa
        st.plotly_chart(fig_pred_multiple)

        # Mostramos coeficiente de correlación
        coef_correl_multiple = np.sqrt(r_squared_multiple)
        st.markdown(f"<p style='color: white;'>Coeficiente de correlación: {coef_correl_multiple:.4f}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: white;'>Por favor, selecciona una variable para el eje Y y al menos una variable para el eje X.</p>", unsafe_allow_html=True)


# Definir funciones de ajuste
def func_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def func_exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def func_inverse(x, a):
    return 1 / a * x

def func_sine(x, a, b):
    return a * np.sin(x) + b

def func_tangent(x, a, b):
    return a * np.tan(x) + b

def func_absolute(x, a, b, c):
    return a * np.abs(x) + b * x + c

def func_polynomial_ratio(x, a, b, c):
    return (a * x**2 + b) / (c * x)

def func_logarithmic(x, a, b):
    return a * np.log(x) + b

def func_linear_product(x, a, b, c):
    return a * x + b * x + c * x

def func_inverse_quadratic(x, a):
    return 1 / a * x**2

def func_inverse_polynomial(x, a, b, c):
    return a / b * x**2 + c * x

# Función para mostrar la ecuación del modelo ajustado
def mostrar_ecuacion(funcion, parametros):
    if funcion == "Función cuadrática":
        return f"y = {parametros[0]:.4f}x² + {parametros[1]:.4f}x + {parametros[2]:.4f}"
    elif funcion == "Función exponencial":
        return f"y = {parametros[0]:.4f}e^({parametros[1]:.4f}x) + {parametros[2]:.4f}"
    elif funcion == "Función inversa":
        return f"y = 1 / ({parametros[0]:.4f}) * x"
    elif funcion == "Función senoidal":
        return f"y = {parametros[0]:.4f}sin(x) + {parametros[1]:.4f}"
    elif funcion == "Función tangencial":
        return f"y = {parametros[0]:.4f}tan(x) + {parametros[1]:.4f}"
    elif funcion == "Función valor absoluto":
        return f"y = {parametros[0]:.4f}|x| + {parametros[1]:.4f}x + {parametros[2]:.4f}"
    elif funcion == "Función cociente entre polinomios":
        return f"y = ({parametros[0]:.4f}x² + {parametros[1]:.4f}) / ({parametros[2]:.4f}x)"
    elif funcion == "Función logarítmica":
        return f"y = {parametros[0]:.4f}ln(x) + {parametros[1]:.4f}"
    elif funcion == "Función lineal con producto de coeficientes":
        return f"y = {parametros[0]:.4f}x + {parametros[1]:.4f}x + {parametros[2]:.4f}x"
    elif funcion == "Función cuadrática inversa":
        return f"y = 1 / ({parametros[0]:.4f}) * x²"
    elif funcion == "Función polinomial inversa":
        return f"y = ({parametros[0]:.4f} / {parametros[1]:.4f}) * x² + {parametros[2]:.4f}x"

# FRAME 5: Regresión No Lineal
if Frames == "Regresión No Lineal":
    st.title("SYDNEY")
    st.header("Regresión No Lineal")
    st.subheader("Variables numéricas")

    # Crear un multiselect para elegir la variable dependiente e independiente
    x_variable = st.sidebar.selectbox('Selecciona la variable independiente (x)', numeric_columns_df1)
    y_variable = st.sidebar.selectbox('Selecciona la variable dependiente (y)', numeric_columns_df1)

    # Asegúrate de que los datos seleccionados no tengan valores NaN
    Vars_Indep = df1[[x_variable]].dropna()
    Var_Dep = df1[y_variable].dropna()

    # Redefinir las variables para el ajuste
    x = Vars_Indep[x_variable].values
    y = Var_Dep.values

    # Crear un selector para elegir la función de ajuste
    funcion_seleccionada = st.sidebar.selectbox('Selecciona la función de ajuste', 
        ["Función cuadrática", "Función exponencial", "Función inversa", 
         "Función senoidal", "Función tangencial", "Función valor absoluto", 
         "Función cociente entre polinomios", "Función logarítmica", 
         "Función lineal con producto de coeficientes", 
         "Función cuadrática inversa", "Función polinomial inversa"])

    # Ajustar el modelo según la función seleccionada
    try:
        if funcion_seleccionada == "Función cuadrática":
            parametros, _ = curve_fit(func_quadratic, x, y)
            yfit = func_quadratic(x, *parametros)
        elif funcion_seleccionada == "Función exponencial":
            try:
                # Proporcionar valores iniciales
                parametros, _ = curve_fit(func_exponential, x, y, p0=[1, 0.1, 1], maxfev=2000)
                yfit = func_exponential(x, *parametros)
            except Exception as e:
                st.error(f"Ocurrió un error con la función exponencial: {e}")
        elif funcion_seleccionada == "Función inversa":
            parametros, _ = curve_fit(func_inverse, x, y)
            yfit = func_inverse(x, *parametros)
        elif funcion_seleccionada == "Función senoidal":
            parametros, _ = curve_fit(func_sine, x, y)
            yfit = func_sine(x, *parametros)
        elif funcion_seleccionada == "Función tangencial":
            parametros, _ = curve_fit(func_tangent, x, y)
            yfit = func_tangent(x, *parametros)
        elif funcion_seleccionada == "Función valor absoluto":
            parametros, _ = curve_fit(func_absolute, x, y)
            yfit = func_absolute(x, *parametros)
        elif funcion_seleccionada == "Función cociente entre polinomios":
            try:
                # Proporcionar valores iniciales
                parametros, _ = curve_fit(func_polynomial_ratio, x, y, p0=[1, 1, 1], maxfev=2000)
                yfit = func_polynomial_ratio(x, *parametros)
            except Exception as e:
                st.error(f"Ocurrió un error con la función cociente entre polinomios: {e}")

        elif funcion_seleccionada == "Función logarítmica":
            try:
                # Proporcionar valores iniciales
                parametros, _ = curve_fit(func_logarithmic, x, y, p0=[1, 1], maxfev=2000)
                yfit = func_logarithmic(x, *parametros)
            except Exception as e:
                st.error(f"Ocurrió un error con la función logarítmica: {e}")

        elif funcion_seleccionada == "Función lineal con producto de coeficientes":
            parametros, _ = curve_fit(func_linear_product, x, y)
            yfit = func_linear_product(x, *parametros)
        elif funcion_seleccionada == "Función cuadrática inversa":
            parametros, _ = curve_fit(func_inverse_quadratic, x, y)
            yfit = func_inverse_quadratic(x, *parametros)
        elif funcion_seleccionada == "Función polinomial inversa":
            parametros, _ = curve_fit(func_inverse_polynomial, x, y)
            yfit = func_inverse_polynomial(x, *parametros)

        # Calcular el coeficiente de determinación R²
        R2 = r2_score(y, yfit)

        # Graficar los datos y el modelo ajustado
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label="Datos originales", color='blue')
        plt.plot(x, yfit, label="Modelo ajustado", color='orange')
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Ajuste de Regresión No Lineal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Mostrar la gráfica en Streamlit
        st.pyplot(plt)

        # Mostrar resultados
        ecuacion = mostrar_ecuacion(funcion_seleccionada, parametros)
        st.markdown(f"<strong style='color: white;'>{funcion_seleccionada}</strong>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white;'>Ecuación del modelo ajustado: {ecuacion}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white;'>Coeficientes: {parametros}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white;'>Coeficiente de determinación (R²): {R2:.4f}</p>", unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
