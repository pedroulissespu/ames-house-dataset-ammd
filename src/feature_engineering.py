"""
Módulo de engenharia de features
"""
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Classe para criação de novas features"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria novas features baseadas no conhecimento do domínio
        """
        df = df.copy()
        
        # 1. Idade da casa
        if 'Year Built' in df.columns and 'Yr Sold' in df.columns:
            df['House_Age'] = df['Yr Sold'] - df['Year Built']
        
        # 2. Anos desde remodelação
        if 'Year Remod/Add' in df.columns and 'Yr Sold' in df.columns:
            df['Years_Since_Remod'] = df['Yr Sold'] - df['Year Remod/Add']
        
        # 3. Total de banheiros
        bathroom_cols = []
        if 'Full Bath' in df.columns:
            bathroom_cols.append('Full Bath')
        if 'Half Bath' in df.columns:
            bathroom_cols.append('Half Bath')
        if 'Bsmt Full Bath' in df.columns:
            bathroom_cols.append('Bsmt Full Bath')
        if 'Bsmt Half Bath' in df.columns:
            bathroom_cols.append('Bsmt Half Bath')
        
        if bathroom_cols:
            df['Total_Bathrooms'] = df[bathroom_cols].sum(axis=1)
        
        # 4. Área total da casa
        area_cols = []
        if 'Gr Liv Area' in df.columns:
            area_cols.append('Gr Liv Area')
        if 'Total Bsmt SF' in df.columns:
            area_cols.append('Total Bsmt SF')
        
        if area_cols:
            df['Total_SF'] = df[area_cols].sum(axis=1)
        
        # 5. Área total de porches
        porch_cols = []
        if 'Wood Deck SF' in df.columns:
            porch_cols.append('Wood Deck SF')
        if 'Open Porch SF' in df.columns:
            porch_cols.append('Open Porch SF')
        if 'Enclosed Porch' in df.columns:
            porch_cols.append('Enclosed Porch')
        if '3Ssn Porch' in df.columns:
            porch_cols.append('3Ssn Porch')
        if 'Screen Porch' in df.columns:
            porch_cols.append('Screen Porch')
        
        if porch_cols:
            df['Total_Porch_SF'] = df[porch_cols].sum(axis=1)
        
        # 6. Indicador de casa remodelada
        if 'Year Built' in df.columns and 'Year Remod/Add' in df.columns:
            df['Is_Remodeled'] = (df['Year Built'] != df['Year Remod/Add']).astype(int)
        
        # 7. Qualidade geral x Condição geral
        if 'Overall Qual' in df.columns and 'Overall Cond' in df.columns:
            df['Overall_Score'] = df['Overall Qual'] * df['Overall Cond']
        
        # 8. Indicador de garagem
        if 'Garage Cars' in df.columns:
            df['Has_Garage'] = (df['Garage Cars'] > 0).astype(int)
        
        # 9. Indicador de piscina
        if 'Pool Area' in df.columns:
            df['Has_Pool'] = (df['Pool Area'] > 0).astype(int)
        
        # 10. Indicador de lareira
        if 'Fireplaces' in df.columns:
            df['Has_Fireplace'] = (df['Fireplaces'] > 0).astype(int)
        
        # 11. Razão área lote / área construída
        if 'Lot Area' in df.columns and 'Gr Liv Area' in df.columns:
            df['Lot_To_Living_Ratio'] = df['Lot Area'] / (df['Gr Liv Area'] + 1)
        
        # 12. Temporada de venda
        if 'Mo Sold' in df.columns:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'
            
            df['Sale_Season'] = df['Mo Sold'].apply(get_season)
        
        print(f"Feats criadas. Shape atual: {df.shape}")
        
        return df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre as variáveis importantes
        """
        df = df.copy()
        
        # Interação entre qualidade e área
        if 'Overall Qual' in df.columns and 'Gr Liv Area' in df.columns:
            df['Qual_Area_Interaction'] = df['Overall Qual'] * df['Gr Liv Area']
        
        # Interação entre idade e qualidade
        if 'House_Age' in df.columns and 'Overall Qual' in df.columns:
            df['Age_Qual_Interaction'] = df['House_Age'] * df['Overall Qual']
        
        return df
    
    @staticmethod
    def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 50) -> list:
        """
        Seleciona as k melhores features usando correlação
        
        Args:
            X: Features
            y: Target
            k: Número de features a selecionar
        
        Returns:
            Lista com nomes das top features
        """
        from sklearn.feature_selection import f_regression, SelectKBest
        
        # Apenas features numéricas
        X_numeric = X.select_dtypes(include=[np.number])
        
        selector = SelectKBest(score_func=f_regression, k=min(k, X_numeric.shape[1]))
        selector.fit(X_numeric, y)
        
        # Obter scores
        scores = pd.DataFrame({
            'feature': X_numeric.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        print("\nTop 10 features por F-score:")
        print(scores.head(10))
        
        return scores.head(k)['feature'].tolist()
