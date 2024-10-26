import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict
import base64
from io import BytesIO

st.set_page_config(page_title="PCA Dashboard", layout="wide")

# Helper Functions
def perform_pca(df: pd.DataFrame, n_components: int, scale: bool = True) -> Dict:
    """
    Perform PCA on the input dataframe
    """
    # Handle scaling
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = df.values
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Prepare results
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.columns
    )
    
    scores = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.index
    )
    
    explained_variance = pd.DataFrame({
        'Eigenvalue': pca.explained_variance_,
        'Proportion': pca.explained_variance_ratio_ * 100,
        'Cumulative': np.cumsum(pca.explained_variance_ratio_) * 100
    })
    
    return {
        'loadings': loadings,
        'scores': scores,
        'explained_variance': explained_variance,
        'pca_obj': pca
    }

def create_scree_plot(explained_variance: pd.DataFrame) -> go.Figure:
    """
    Create scree plot using plotly
    """
    fig = go.Figure()
    
    # Add variance bars
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_variance))],
        y=explained_variance['Proportion'],
        name='Explained Variance',
        text=explained_variance['Proportion'].round(2).astype(str) + '%',
        textposition='auto',
    ))
    
    # Add cumulative line
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(explained_variance))],
        y=explained_variance['Cumulative'],
        name='Cumulative Variance',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Scree Plot',
        xaxis_title='Principal Components',
        yaxis_title='Explained Variance (%)',
        yaxis2=dict(
            title='Cumulative Variance (%)',
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )
    
    return fig

def create_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, pc1: int, pc2: int) -> go.Figure:
    """
    Create biplot using plotly
    """
    fig = go.Figure()
    
    # Plot scores
    fig.add_trace(go.Scatter(
        x=scores.iloc[:, pc1],
        y=scores.iloc[:, pc2],
        mode='markers+text',
        text=scores.index,
        textposition='top center',
        name='Observations',
        textfont=dict(size=8),
    ))
    
    # Plot loadings
    for i, (idx, row) in enumerate(loadings.iloc[:, [pc1, pc2]].iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row[0]],
            y=[0, row[1]],
            mode='lines+text',
            name=idx,
            text=[None, idx],
            textposition='top right',
            line=dict(color='red'),
            textfont=dict(size=8),
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Biplot (PC{pc1+1} vs PC{pc2+1})',
        xaxis_title=f'PC{pc1+1}',
        yaxis_title=f'PC{pc2+1}',
        showlegend=False
    )
    
    return fig

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a download link for a dataframe
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Main App
def main():
    st.title("Principal Component Analysis Dashboard")
    
    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file, index_col=0)
        
        # Select variables
        st.sidebar.header("Variable Selection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_vars = st.sidebar.multiselect(
            "Select variables for PCA",
            options=list(numeric_cols),
            default=list(numeric_cols)
        )
        
        if len(selected_vars) < 2:
            st.error("Please select at least 2 variables for PCA")
            return
        
        # PCA options
        st.sidebar.header("PCA Options")
        scale_data = st.sidebar.checkbox("Scale data", value=True)
        n_components = st.sidebar.slider(
            "Number of components",
            min_value=2,
            max_value=len(selected_vars),
            value=min(len(selected_vars), 5)
        )
        
        # Perform PCA
        pca_results = perform_pca(df[selected_vars], n_components, scale_data)
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "Scree Plot",
            "Loadings",
            "Scores",
            "Biplot"
        ])
        
        # Tab 1: Scree Plot
        with tab1:
            st.plotly_chart(
                create_scree_plot(pca_results['explained_variance']),
                use_container_width=True
            )
            st.markdown(get_download_link(
                pca_results['explained_variance'],
                'explained_variance.csv'
            ), unsafe_allow_html=True)
        
        # Tab 2: Loadings
        with tab2:
            st.dataframe(pca_results['loadings'].style.format("{:.3f}"))
            st.markdown(get_download_link(
                pca_results['loadings'],
                'loadings.csv'
            ), unsafe_allow_html=True)
        
        # Tab 3: Scores
        with tab3:
            st.dataframe(pca_results['scores'].style.format("{:.3f}"))
            st.markdown(get_download_link(
                pca_results['scores'],
                'scores.csv'
            ), unsafe_allow_html=True)
        
        # Tab 4: Biplot
        with tab4:
            col1, col2 = st.columns(2)
            pc1 = col1.selectbox("X-axis", range(n_components), 0)
            pc2 = col2.selectbox("Y-axis", range(n_components), 1)
            
            st.plotly_chart(
                create_biplot(
                    pca_results['scores'],
                    pca_results['loadings'],
                    pc1,
                    pc2
                ),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
